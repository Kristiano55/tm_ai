import numpy as np, os
# TICKGYM_FAKE_DETOUR=1: at z=200 the car leaves the straight and drives a 620 m side loop (out to x=310 and back,
# almost sideways) before going on, so a straight reference line "cuts the corner" its route takes. >40 m off a
# straight line for ~6.8 s at full speed: just inside the 8 s "far off line" limit, any lifting fails.
# TICKGYM_FAKE_DETOUR=<file path>: the loop exists only while that file exists (checked every step).
# TICKGYM_FAKE_HANG=<file path>: when that file exists, the next car to step deletes it and hangs forever.
_DET=os.environ.get("TICKGYM_FAKE_DETOUR","")
_HANG=os.environ.get("TICKGYM_FAKE_HANG","")
def _detour(): return _DET=="1" or (_DET not in ("","0") and os.path.exists(_DET))
# TICKGYM_FAKE_MS=5: each step waits ~5 ms (random, like a busy TICK server); a raycast ~1 ms. Default 0 = instant.
import time, random
FAKE_MS=float(os.environ.get("TICKGYM_FAKE_MS","0"))
def _wait(ms):
    if ms>0: time.sleep(random.expovariate(1.0/ms)/1000.0)
def _path(d):
    if not _detour() or d<200: return [0.,0.,d]
    a,b,c=np.array([0.,0.,200.]),np.array([310.,0.,202.]),np.array([0.,0.,204.])
    l1,l2=np.linalg.norm(b-a),np.linalg.norm(c-b)
    if d<200+l1: return list(a+(b-a)*(d-200)/l1)
    if d<200+l1+l2: return list(b+(c-b)*(d-200-l1)/l2)
    return [0.,0.,204.+d-200-l1-l2]
from types import SimpleNamespace as NS
class TransportError(Exception): pass
class TelemetryGroup:
    RACE=1;VEHICLE=2;DRIVETRAIN=4;WHEELS=8;EFFECTS=16;RUNTIME=32
class Ray:
    def __init__(s,r): s.range_m=r
class Lidar:
    def __init__(s,rays): s.rays=rays
    @staticmethod
    def fan(rays,fov_degrees,range_m): return Lidar([Ray(range_m) for _ in range(rays)])
    @staticmethod
    def directions(a,b,range_m): return Lidar([Ray(range_m) for _ in a])
class Action:
    def __init__(s,tick,kind,value=0): s.tick,s.kind,s.value=tick,kind,value
class InputText:
    @staticmethod
    def write(p,a): open(p,"w").write("x")
    @staticmethod
    def read(p):
        return [Action(int(l.split()[0])//10,l.split()[1],int(l.split()[2])) for l in open(p) if l.strip()]
class G:
    def __init__(s,lid):
        s.n=len(lid.rays) if lid else 0
        s.map=NS(waypoints=[NS(group_id=1,vertices=[[0,0,500]]),NS(group_id=2,vertices=[[0,0,1000]])],start_position=[0,0,0],lap_count=1)
    def tel(s):
        sp=s.v
        p=_path(s.z); z=p[2]
        return NS(vehicle=NS(speed=sp,linear_velocity=[0,0,sp],angular_velocity=[0,0,0],rotation=[0,0,0,1],position=p,in_water=0),
          wheels=NS(contact=[1]*4,slipping=[0]*4,angular_velocity=[10]*4,icing_percent=[0]*4),
          drivetrain=NS(steer=0,gear=3),race=NS(checkpoints_taken=int(z//500),race_tick=s.tick,race_time_ms=s.tick*10,entered_off_zone=False),
          runtime=NS(wall_hit_this_step=0,wetness=0),effects=NS(turbo=0),
          lidar=NS(hit=[0]*s.n,distance=[0]*s.n) if s.n else None,finished=z>=1000)
    acc=True
    @property
    def next_tick(s): return s.tick
    def reset(s,seed=0): s.z=0.;s.v=0.;s.tick=0;s.acc=True;return s.tel()
    def step(s,ticks,actions=(),steer=None,accelerate=None,brake=None):
        for a in actions:
            if a.kind=='accel': s.acc=bool(a.value)
        if accelerate is not None: s.acc=accelerate
        accelerate=s.acc
        if _HANG and os.path.exists(_HANG):
            try: os.remove(_HANG); time.sleep(10**6)
            except FileNotFoundError: pass
        _wait(FAKE_MS); s.tick+=ticks; s.v=min(s.v+(1.6 if accelerate else -1),80); s.v=max(s.v,0); s.z+=s.v*ticks/100; return s.tel()
    def raycast(s,rays):
        _wait(FAKE_MS/5); return NS(hit=[0]*len(rays),distance=[0]*len(rays))
    info=NS(max_snapshots_per_session=64)
    def snapshot(s):
        G.snaps=getattr(G,'snaps',{}); i=len(G.snaps)+1; G.snaps[i]=(s.z,s.v,s.tick); return NS(id=i,tick=s.tick)
    def restore(s,snap):
        s.z,s.v,s.tick=G.snaps[snap.id]; return s.tel()
    def discard(s,*a): pass
    def export(s): return NS(finish_ms=s.tick*10,actions=[])
    def close(s): pass
class Client:
    def __init__(s,*a,**k): pass
    def open_map(s,m): return NS(connect=lambda g,l,timeout=None:G(l),close=lambda:None)
