"""Report the raw BVH Euler channels of the leg joints, so the mapping onto the gym
Humanoid's 17 hinge DOFs can be argued from data instead of guessed.

Position-based joint angles (as in bvh_gait_stats.py) conflate the pelvis/lateral offsets
with true hip abduction, so for the frontal-plane question the Euler channels themselves are
the honest quantity: the robot's hip_x/hip_y/hip_z, knee, and the missing ankle each have to
be identified with one of these channels (or declared unrepresentable).

Usage: python3 probe_bvh_euler.py <clip.bvh> [...]
"""
import sys
import numpy as np
import bvh_gait_stats as B

WANT = ["LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot", "Hips",
        "LowerBack", "Spine", "Spine1"]


def report(path):
    joints, n, dt, data = B.parse(path)
    names = [j["name"] for j in joints]
    ix = {k: i for i, k in enumerate(names)}
    if path.split("/")[-1].startswith("cmu_"):
        data = data[1:]
    print(f"\n=== {path.split('/')[-1]} ===")
    # hips translation channels tell us whether root motion is present
    if data.shape[1] >= 6 and joints[0]["channels"][0] == "Xposition":
        t = data[:, :3]
        print(f"  root translation range: x={np.ptp(t[:,0]):.2f} y={np.ptp(t[:,1]):.2f} "
              f"z={np.ptp(t[:,2]):.2f} (units)  ->  root motion "
              f"{'PRESENT' if np.ptp(t[:,[0,2]].max()) > 5 or np.ptp(t[:,[0,2]].max())*0.06 > 2 else 'check scale'}")
    for nm in WANT:
        if nm not in ix:
            continue
        j = joints[ix[nm]]
        rot_cols = [c for c in j["channels"] if c.endswith("rotation")]
        if not rot_cols:
            continue
        s = j["start"] + len([c for c in j["channels"] if c.endswith("position")])
        vals = data[:, s: s + len(rot_cols)]
        parts = [f"{c[0]}: p5={np.percentile(vals[:, k],5):7.1f} p50={np.median(vals[:, k]):7.1f} "
                 f"p95={np.percentile(vals[:, k],95):7.1f} rng={np.ptp(vals[:, k]):6.1f}"
                 for k, c in enumerate(rot_cols)]
        print(f"  {nm:11s} " + " | ".join(parts))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
