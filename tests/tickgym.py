import numpy as np, os
# TICKGYM_FAKE_DETOUR=1: at z=200 the car leaves the straight and drives a 620 m side loop (out to x=310 and back,
# almost sideways) before going on, so a straight reference line "cuts the corner" its route takes. >40 m off a
# straight line for ~6.8 s at full speed: just inside the 8 s "far off line" limit, any lifting fails.
# TICKGYM_FAKE_DETOUR=<file path>: the loop exists only while that file exists (checked every step).
# TICKGYM_FAKE_HANG=<file path>: when that file exists, the next car to step deletes it and hangs forever.
_DET=os.environ.get("TICKGYM_FAKE_DETOUR","")
_HANG=os.environ.get("TICKGYM_FAKE_HANG","")
# TICKGYM_FAKE_FAIL=<file path>: when that file exists, the next car to reset deletes it and raises TransportError
# (like "the connection failed: timed out"); the file content = how many resets in a row fail (2 kills a car)
_FAIL=os.environ.get("TICKGYM_FAKE_FAIL","")
# TICKGYM_FAKE_WALL=<z>: a wall across the road at z (e.g. 150): crossing it faster than 20 m/s wrecks the car (speed 0
# for the rest of the lap = "stuck"); only braking to <= 20 m/s before it gets through (a hard spot after the start)
WALL=float(os.environ.get("TICKGYM_FAKE_WALL","0") or 0)
# Surfaces: asphalt (material 16) everywhere except dirt (material 6) for z 600..800; gameplay id 3 ("turbo")
# for z 300..320, else 0. Ice fields are 0 (a summer map). The 4 downward rays (32..35) hit the road ~2 m below
# and see the material ~20 m ahead.
ASPHALT, DIRT = 16, 6
def _mat(z): return DIRT if 600<=z<800 else ASPHALT
def _gp(z): return 3 if 300<=z<320 else 0
# TICKGYM_FAKE_CORNER=1: a "corner" for z 400..460: above 35 m/s the car is slowed to 35, unless gas AND brake are
# held together (a drift), which keeps up to 60 m/s. So gas+brake is the fast way through it.
CORNER=os.environ.get("TICKGYM_FAKE_CORNER")=="1"
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
    def write(p,a): open(p,"w").write("".join(f"{int(x.tick)*10} {x.kind} {int(x.value)}\n" for x in a))
    @staticmethod
    def read(p):
        return [Action(int(l.split()[0])//10,l.split()[1],int(l.split()[2])) for l in open(p) if l.strip()]
# Map "fake:<length>:<surface>" (e.g. fake:600:dirt): a straight of that length, checkpoint halfway, the whole road
# one surface. Surfaces: asphalt (material 16, top speed 80 m/s), dirt (6, 50 m/s), grass (2, 45 m/s). Any other
# map name = the default 1000 m map above (asphalt with the dirt stretch and the turbo strip).
SURF={"asphalt":(16,80.0),"dirt":(6,50.0),"grass":(2,45.0)}
class G:
    def __init__(s,lid,m="x"):
        s.n=len(lid.rays) if lid else 0
        s.L,s.surf=1000.0,None
        if str(m).startswith("fake:"):
            parts=str(m).split(":"); s.L=float(parts[1]); s.surf=parts[2] if len(parts)>2 else "asphalt"
        s.vmax=SURF[s.surf][1] if s.surf else 80.0
        wps=[NS(group_id=1,vertices=[[0,0,s.L/2]],tag="Checkpoint"),NS(group_id=2,vertices=[[0,0,s.L]],tag="Finish")]
        if os.environ.get("TICKGYM_FAKE_STARTWP"):    # the start block listed as a waypoint group (like real maps)
            wps.insert(0,NS(group_id=0,vertices=[[0,0,8.0]],tag=os.environ["TICKGYM_FAKE_STARTWP"]))
        s.map=NS(waypoints=wps,start_position=[0,0,0],lap_count=1)
    def mat(s,z): return SURF[s.surf][0] if s.surf else _mat(z)
    def gp(s,z): return 0 if s.surf else _gp(z)
    def tel(s):
        sp=s.v
        p=_path(s.z); z=p[2]
        return NS(vehicle=NS(speed=sp,linear_velocity=[0,0,sp],angular_velocity=[0,0,0],rotation=[0,0,0,1],position=p,in_water=0),
          wheels=NS(contact=[1]*4,slipping=[0]*4,angular_velocity=[10]*4,icing_percent=[0]*4,material_id=[s.mat(z)]*4),
          drivetrain=NS(steer=0,gear=3),race=NS(checkpoints_taken=int(z//(s.L/2)),race_tick=s.tick,race_time_ms=s.tick*10,entered_off_zone=False),
          runtime=NS(wall_hit_this_step=0,wetness=0,**{f"{w}_surface_gameplay_id":float(s.gp(z)) for w in ("fl","fr","rl","rr")},
                     **{f"{w}_icing_{k}":0.0 for w in ("fl","fr","rl","rr") for k in ("state","blend")},
                     front_wheels_icing_grip_scale=1.0,rear_wheels_icing_grip_scale=1.0,is_any_wheel_on_ice_or_snow=0.0),effects=NS(turbo=0),
          lidar=s._scan(s.n,z) if s.n else None,finished=z>=s.L)
    acc=True
    @property
    def next_tick(s): return s.tick
    def reset(s,seed=0):
        if _FAIL and os.path.exists(_FAIL):          # file content = how many resets fail in a row (default 1)
            if open(_FAIL).read().strip()=="1455":   # Windows out of virtual memory (paging file too small)
                raise OSError("[WinError 1455] The paging file is too small for this operation to complete")
            try:
                n=int(open(_FAIL).read().strip() or 1)
                if n>1: open(_FAIL,"w").write(str(n-1))
                else: os.remove(_FAIL)
            except (OSError,ValueError): pass
            else: raise TransportError("the connection failed: timed out")
        s.z=0.;s.v=0.;s.tick=0;s.acc=True;s.brk=False;s.rec=[];s.crashed=False;return s.tel()
    def step(s,ticks,actions=(),steer=None,accelerate=None,brake=None):
        for a in actions:
            if a.kind=='accel': s.acc=bool(a.value)
            if a.kind=='brake': s.brk=bool(a.value)
        if accelerate is not None:
            if bool(accelerate)!=s.acc: s.rec.append(Action(s.tick,'accel',int(bool(accelerate))))   # what export() returns
            s.acc=bool(accelerate)
        if brake is not None:
            if bool(brake)!=getattr(s,'brk',False): s.rec.append(Action(s.tick,'brake',int(bool(brake))))
            s.brk=bool(brake)
        accelerate=s.acc
        if _HANG and os.path.exists(_HANG):
            try: os.remove(_HANG); time.sleep(10**6)
            except FileNotFoundError: pass
        _wait(FAKE_MS); s.tick+=ticks; s.v=min(s.v+(1.6 if accelerate else -1),s.vmax); s.v=max(s.v,0)
        if getattr(s,'brk',False) and not accelerate: s.v=max(s.v-2,0)
        if CORNER and 400<=s.z<460: s.v=min(s.v,60.0 if (accelerate and s.brk) else 35.0)
        if getattr(s,'crashed',False): s.v=0.0             # TICKGYM_FAKE_WALL: wrecked against the wall for good
        nz=s.z+s.v*ticks/100
        if WALL and s.z<WALL<=nz and s.v>20.0:              # hit the wall faster than 20 m/s: stopped dead
            s.crashed,s.v,nz=True,0.0,WALL-1.0
        s.z=nz; return s.tel()
    def _scan(s,n,z):
        hit=[1 if 32<=i<36 else 0 for i in range(n)]
        return NS(hit=hit,distance=[2.0 if h else 0 for h in hit],material_id=[s.mat(z+20) if h else 0 for h in hit])
    def raycast(s,rays):
        _wait(FAKE_MS/5); return s._scan(len(rays),_path(s.z)[2])
    info=NS(max_snapshots_per_session=64)
    def snapshot(s):
        G.snaps=getattr(G,'snaps',{}); i=len(G.snaps)+1; G.snaps[i]=(s.z,s.v,s.tick); return NS(id=i,tick=s.tick)
    def restore(s,snap):
        s.z,s.v,s.tick=G.snaps[snap.id]; s.crashed=False; return s.tel()
    def discard(s,*a): pass
    def export(s): return NS(finish_ms=s.tick*10,actions=list(getattr(s,"rec",[])))
    def close(s): pass
class Client:
    def __init__(s,*a,**k): pass
    def open_map(s,m):
        import hashlib
        uid="FAKE"+hashlib.sha1(str(m).encode()).hexdigest()[:10]
        return NS(connect=lambda g,l,timeout=None:G(l,m),close=lambda:None,
                  info=NS(map=NS(map_uid=uid,sha256=hashlib.sha256(str(m).encode()).hexdigest())))
