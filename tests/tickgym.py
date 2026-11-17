import numpy as np, os
# TICKGYM_FAKE_DETOUR=1: at z=200 the car leaves the straight and drives a 600 m side loop (out to x=300 and back, almost sideways)
# before going on, so a straight reference line "cuts the corner" its route takes.
DETOUR=os.environ.get("TICKGYM_FAKE_DETOUR")=="1"
def _path(d):
    if not DETOUR or d<200: return [0.,0.,d]
    a,b,c=np.array([0.,0.,200.]),np.array([300.,0.,202.]),np.array([0.,0.,204.])
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
        s.n=len(lid.rays)
        s.map=NS(waypoints=[NS(group_id=1,vertices=[[0,0,500]]),NS(group_id=2,vertices=[[0,0,1000]])],start_position=[0,0,0],lap_count=1)
    def tel(s):
        sp=s.v
        p=_path(s.z); z=p[2]
        return NS(vehicle=NS(speed=sp,linear_velocity=[0,0,sp],angular_velocity=[0,0,0],rotation=[0,0,0,1],position=p,in_water=0),
          wheels=NS(contact=[1]*4,slipping=[0]*4,angular_velocity=[10]*4,icing_percent=[0]*4),
          drivetrain=NS(steer=0,gear=3),race=NS(checkpoints_taken=int(z//500),race_tick=s.tick,race_time_ms=s.tick*10,entered_off_zone=False),
          runtime=NS(wall_hit_this_step=0,wetness=0),effects=NS(turbo=0),
          lidar=NS(hit=[0]*s.n,distance=[0]*s.n),finished=z>=1000)
    acc=True
    @property
    def next_tick(s): return s.tick
    def reset(s,seed=0): s.z=0.;s.v=0.;s.tick=0;s.acc=True;return s.tel()
    def step(s,ticks,actions=(),steer=None,accelerate=None,brake=None):
        for a in actions:
            if a.kind=='accel': s.acc=bool(a.value)
        if accelerate is not None: s.acc=accelerate
        accelerate=s.acc
        s.tick+=ticks; s.v=min(s.v+(1.6 if accelerate else -1),80); s.v=max(s.v,0); s.z+=s.v*ticks/100; return s.tel()
    info=NS(max_snapshots_per_session=64)
    def snapshot(s):
        G.snaps=getattr(G,'snaps',{}); i=len(G.snaps)+1; G.snaps[i]=(s.z,s.v,s.tick); return NS(id=i,tick=s.tick)
    def restore(s,snap):
        s.z,s.v,s.tick=G.snaps[snap.id]; return s.tel()
    def discard(s,*a): pass
    def export(s): return NS(finish_ms=s.tick*10,actions=[])
    def close(s): pass
class Client:
    def open_map(s,m): return NS(connect=lambda g,l:G(l),close=lambda:None)
