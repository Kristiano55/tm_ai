"""
PPO extensions (an importable module, so saved models reference ppo_ext.* and load in every process):

  * PrivPolicy  - asymmetric critic: the last N_PRIV inputs (privileged, appended after the lidar) reach only the
                  value net; the actor sees them as 0, so its weights for them never change from 0. Plus aux_head
                  (3 outputs on the actor's latent, training only: never used to choose actions).
  * TMPPO       - PPO whose train() runs the normal PPO update (target_kl etc. unchanged) and then a few extra
                  gradient steps for self-imitation (Oh et al. 2018) and the auxiliary heads.
  * SILBuffer   - the AI's own top-20 finished laps per map (obs, action, raw reward), saved as sil.npz in the run.
  * aux_labels  - targets of the aux heads from a rollout: wall hit within 0.5 s, wheels grounded within 0.3 s,
                  seconds to the next checkpoint.
"""
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor

N_PRIV = 6            # ghost time left, PB time left, to next CP, to finish, ghost gap, section failure rate
AUX_HIT_S, AUX_GROUND_S, AUX_CP_CAP_S = 0.5, 0.3, 10.0
SIL_LAPS, SIL_BATCH, EXTRA_STEPS, AUX_BATCH = 20, 256, 4, 512


class MaskedMlpExtractor(MlpExtractor):
    def __init__(self, *a, n_priv=N_PRIV, **k):
        super().__init__(*a, **k)
        self.n_priv = n_priv

    def forward_actor(self, features):
        if self.n_priv:
            features = th.cat([features[..., :-self.n_priv], th.zeros_like(features[..., -self.n_priv:])], -1)
        return self.policy_net(features)


class PrivPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = MaskedMlpExtractor(self.features_dim, net_arch=self.net_arch,
                                                activation_fn=self.activation_fn, device=self.device)
        self.aux_head = nn.Linear(self.mlp_extractor.latent_dim_pi, 3)     # hit / grounded logits, CP time


STEER_T = 0.01                      # temperature of the level-expectation steer mean (low: matches the old argmax)
A_MAX = 0.999                       # |tanh| limit of steer mean and log_prob (0.999 x 127 still rounds to 127)
STEER_STD0 = 6.0 / 127.0            # starting steer exploration (std after tanh, near the centre)


class HybridDist:
    """Action [steer, mode]: steer in [-1, 1] = tanh(Gaussian u) (x127 -> TM steering), mode = Categorical.
    log_prob includes the tanh correction; entropy = Categorical + Gaussian (of u)."""
    EPS = 1e-6

    def __init__(self, logits, mu, log_std):
        self.cat = th.distributions.Categorical(logits=logits)
        self.mu, self.std = mu, log_std.exp().expand_as(mu)
        self.normal = th.distributions.Normal(mu, self.std)

    def log_prob(self, actions):
        a = actions[..., 0].clamp(-A_MAX, A_MAX)           # full lock (+-127) = the mean's own limit
        u = th.atanh(a)
        lp = self.normal.log_prob(u) - th.log(1 - a * a + self.EPS)
        return lp + self.cat.log_prob(actions[..., 1].round().long().clamp(0, self.cat.logits.shape[-1] - 1))

    def entropy(self):
        return self.cat.entropy() + self.normal.entropy()

    def sample(self):
        return th.stack([th.tanh(self.normal.rsample()), self.cat.sample().float()], -1)

    def mode(self):
        return th.stack([th.round(th.tanh(self.mu) * 127.0) / 127.0, self.cat.probs.argmax(-1).float()], -1)

    def get_actions(self, deterministic=False):
        return self.mode() if deterministic else self.sample()


class AnalogPolicy(PrivPolicy):
    """PrivPolicy with analog steering. action_net keeps the discrete layout (steer-level logits + mode logits),
    so old runs convert by plain copy: steer mean u = atanh(sum softmax(level logits / T) * level / 127) + steer_res,
    steer_res starts at 0 (= the old argmax). Learnable log_std, capped by the exploration decay."""

    def __init__(self, *a, steer_levels=None, **k):
        self.steer_levels = list(steer_levels or [])
        super().__init__(*a, **k)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        n_lv, n_modes = len(self.steer_levels), int(round(float(self.action_space.high[1]))) + 1
        lat = self.mlp_extractor.latent_dim_pi
        self.action_net = nn.Linear(lat, n_lv + n_modes)
        nn.init.orthogonal_(self.action_net.weight, gain=0.01)
        nn.init.zeros_(self.action_net.bias)
        self.steer_res = nn.Linear(lat, 1)
        nn.init.zeros_(self.steer_res.weight)
        nn.init.zeros_(self.steer_res.bias)
        self.log_std = nn.Parameter(th.tensor([float(np.log(STEER_STD0))]))
        self.register_buffer("lv", th.tensor(np.asarray(self.steer_levels, np.float32) / 127.0))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi):
        out = self.action_net(latent_pi)
        n_lv = len(self.steer_levels)
        e = (th.softmax(out[..., :n_lv] / STEER_T, -1) * self.lv).sum(-1).clamp(-A_MAX, A_MAX)
        mu = th.atanh(e) + self.steer_res(latent_pi)[..., 0]
        return HybridDist(out[..., n_lv:], mu, self.log_std)

    def _predict(self, observation, deterministic=False):
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def _get_constructor_parameters(self):
        d = super()._get_constructor_parameters()
        d["steer_levels"] = self.steer_levels
        return d

    def steer_std(self):
        return float(self.log_std.detach().exp()) * 127.0


def aux_labels(rows, starts, ticks_per_step):
    """rows [T, E, 4] = (wall hit, grounded, race tick, checkpoints) after each step; starts [T, E] = the obs is a
    new episode's first. Returns y [T, E, 3] and mask [T, E, 3] (CP time only where the next CP is in sight)."""
    T, E = rows.shape[:2]
    kh = max(1, int(round(AUX_HIT_S * 100 / ticks_per_step)))
    kg = max(1, int(round(AUX_GROUND_S * 100 / ticks_per_step)))
    y, m = np.zeros((T, E, 3), np.float32), np.zeros((T, E, 3), np.float32)
    for e in range(E):
        for t in range(T):
            hit = grd = 0.0
            cp = None
            for j in range(t, T):
                if j > t and starts[j, e]:                       # next episode: stop looking
                    break
                if j - t < kh and rows[j, e, 0] > 0:
                    hit = 1.0
                if j - t < kg and rows[j, e, 1] > 0:
                    grd = 1.0
                if cp is None and j > t and rows[j, e, 3] > rows[t, e, 3]:
                    cp = (rows[j, e, 2] - rows[t, e, 2]) / 100.0
                if j - t >= max(kh, kg) and (cp is not None or j - t > AUX_CP_CAP_S * 100 / ticks_per_step):
                    break
            full = t + max(kh, kg) <= T                          # window not cut by the rollout end
            y[t, e, 0], y[t, e, 1] = hit, grd
            m[t, e, 0] = m[t, e, 1] = 1.0 if (full or hit) else 0.0
            if cp is not None:
                y[t, e, 2], m[t, e, 2] = min(cp, AUX_CP_CAP_S) / AUX_CP_CAP_S, 1.0
    return y, m


class SILBuffer:
    """Top laps per map (fastest first). Raw (unscaled) rewards: returns are scaled with the CURRENT reward scaling
    at training time, so old laps stay on the same scale as the value net."""

    def __init__(self, analog=False, levels=None):
        self.analog, self.levels = analog, np.asarray(levels if levels is not None else [], np.float64)
        self.laps = {}                                            # map key -> [(ms, obs f16 [T,D], act [T,2], rew [T])]

    def add(self, key, ms, obs, act, rew):
        L = self.laps.setdefault(key or "", [])
        if len(L) >= SIL_LAPS and ms >= L[-1][0]:
            return False
        L.append((int(ms), np.asarray(obs, np.float16), np.asarray(act, np.float32 if self.analog else np.int16), np.asarray(rew, np.float32)))
        L.sort(key=lambda x: x[0])
        del L[SIL_LAPS:]
        return True

    def n(self):
        return sum(len(o) for L in self.laps.values() for _, o, _, _ in L)

    def sample(self, k, gamma, scale, clip=10.0):
        items = [x for L in self.laps.values() for x in L]
        if not items:
            return None
        obs, act, ret = [], [], []
        for ms, o, a, r in items:                                 # discounted returns (finished lap: no bootstrap)
            r = np.clip(r * scale, -clip, clip)                   # exactly as PPO sees rewards (VecNormalize)
            g, G = 0.0, np.empty(len(r), np.float32)
            for i in range(len(r) - 1, -1, -1):
                g = r[i] + gamma * g
                G[i] = g
            obs.append(o)
            act.append(a)
            ret.append(G)
        O, A, R = np.concatenate(obs), np.concatenate(act), np.concatenate(ret)
        idx = np.random.randint(0, len(O), size=min(k, len(O)))
        return O[idx].astype(np.float32), A[idx].astype(np.float32 if self.analog else np.int64), R[idx]

    def save(self, path):
        keys, ms, lens, O, A, R = [], [], [], [], [], []
        for k, L in self.laps.items():
            for m, o, a, r in L:
                keys.append(k)
                ms.append(m)
                lens.append(len(o))
                O.append(o)
                A.append(a)
                R.append(r)
        if not O:
            return
        tmp = f"{path}.tmp.npz"
        np.savez_compressed(tmp, keys=np.array(keys, dtype="U80"), ms=np.array(ms), lens=np.array(lens),
                            obs=np.concatenate(O), act=np.concatenate(A).astype(np.float32), analog=np.array(self.analog), rew=np.concatenate(R))
        import os
        os.replace(tmp, path)

    def load(self, path, dim):
        try:
            with np.load(path, allow_pickle=False) as z:
                if z["obs"].shape[1] != dim:
                    return False                                  # older input layout: start a new collection
                act = np.asarray(z["act"], np.float64)
                was = bool(z["analog"]) if "analog" in z.files else False
                if len(self.levels) and was != self.analog:          # convert stored steering to the current kind
                    act = act.copy()
                    if self.analog:                                   # level index -> steer value / 127
                        act[:, 0] = self.levels[np.clip(act[:, 0].astype(int), 0, len(self.levels) - 1)] / 127.0
                    else:                                             # steer value -> nearest level
                        act[:, 0] = np.abs(act[:, :1] * 127.0 - self.levels[None]).argmin(1)
                i = 0
                for k, m, n in zip(z["keys"], z["ms"], z["lens"]):
                    self.add(str(k), int(m), z["obs"][i:i + n], act[i:i + n], z["rew"][i:i + n])
                    i += int(n)
            return True
        except (OSError, ValueError, KeyError):
            return False


class TMPPO(PPO):
    sil_weight, aux_weight = 0.1, 0.05
    sil, aux_rows, ret_scale, ret_clip = None, None, 1.0, 10.0

    def train(self):
        super().train()                                           # the normal PPO update (target_kl, clip, ...)
        if not (self.sil_weight > 0 or self.aux_weight > 0):
            return
        pol, dev = self.policy, self.device
        pol.set_training_mode(True)
        aux = None
        if self.aux_weight > 0 and self.aux_rows is not None and len(self.aux_rows) == self.n_steps:
            rb = self.rollout_buffer
            y, m = aux_labels(np.asarray(self.aux_rows), rb.episode_starts, getattr(self, "ticks_per_step", 5))
            aux = (rb.observations.reshape(-1, rb.observations.shape[-1]), y.reshape(-1, 3), m.reshape(-1, 3))
        sil_l, aux_l, sil_frac = [], [], []
        for _ in range(EXTRA_STEPS):
            loss = 0.0
            if aux is not None:
                idx = np.random.randint(0, len(aux[0]), AUX_BATCH)
                obs = th.as_tensor(aux[0][idx], device=dev).float()
                yy, mm = th.as_tensor(aux[1][idx], device=dev), th.as_tensor(aux[2][idx], device=dev)
                f = pol.extract_features(obs)
                f = f[0] if isinstance(f, tuple) else f
                out = pol.aux_head(pol.mlp_extractor.forward_actor(f))
                bce = F.binary_cross_entropy_with_logits(out[:, :2], yy[:, :2], reduction="none")
                mse = (out[:, 2] - yy[:, 2]) ** 2
                la = ((bce * mm[:, :2]).sum() + (mse * mm[:, 2]).sum()) / mm.sum().clamp(min=1.0)
                loss = loss + self.aux_weight * la
                aux_l.append(float(la))
            if self.sil_weight > 0 and self.sil is not None:
                s = self.sil.sample(SIL_BATCH, self.gamma, self.ret_scale, self.ret_clip)
                if s is not None:
                    obs, act, ret = (th.as_tensor(x, device=dev) for x in s)
                    values, logp, _ = pol.evaluate_actions(obs.float(), act)
                    adv = (ret.float() - values.flatten()).clamp(min=0.0)          # only where R > V(s)
                    ls = -(logp * adv.detach()).mean() + 0.5 * 0.01 * (adv ** 2).mean()
                    loss = loss + self.sil_weight * ls
                    sil_l.append(float(ls))
                    sil_frac.append(float((adv > 0).float().mean()))
            if isinstance(loss, float):
                break
            pol.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(pol.parameters(), self.max_grad_norm)
            pol.optimizer.step()
        if aux_l:
            self.logger.record("train/aux_loss", float(np.mean(aux_l)))
        if sil_l:
            self.logger.record("train/sil_loss", float(np.mean(sil_l)))
            self.logger.record("train/sil_active", float(np.mean(sil_frac)))
            self.logger.record("train/sil_steps", float(self.sil.n()))
