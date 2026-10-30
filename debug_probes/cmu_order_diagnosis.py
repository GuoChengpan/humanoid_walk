"""Diagnose the CMU (cgspeed/MotionBuilder-friendly) BVH rotation composition order.

The una-dinosauria CMU BVH files FK into a body whose hips end up 0.15 above the toe while
the thigh+shin offsets sum to ~13 -- so either the offsets and the root translation are in
different units, or the Euler order is not the one LAFAN1 needs. This tries all 6 orders and
scores each by how "unfolded" the leg chain is: max over frames of |hips - toe| / (thigh+shin).
"""
import sys
import itertools
import numpy as np
import bvh_gait_stats as B


def fk_order(joints, data, order):
    n, J = data.shape[0], len(joints)
    pos = np.zeros((n, J, 3))
    rotm = np.zeros((n, J, 3, 3))
    for f in range(n):
        row = data[f]
        for k, j in enumerate(joints):
            ch = j["channels"]
            vals = row[j["start"]: j["start"] + len(ch)]
            d = dict(zip([c[0] for c in ch if c.endswith("rotation")],
                         [v for c, v in zip(ch, vals) if c.endswith("rotation")]))
            R = np.eye(3)
            for a in order:
                if a in d:
                    R = R @ B.rot(a, d[a])
            if ch and ch[0].endswith("position"):
                p, par_R, par_p = vals[:3], np.eye(3), np.zeros(3)
            else:
                p = j["offset"]
                par_R, par_p = rotm[f, j["parent"]], pos[f, j["parent"]]
            pos[f, k] = par_p + par_R @ p
            rotm[f, k] = par_R @ R
    return pos, rotm


def score(path, step=7):
    joints, n, dt, data = B.parse(path)
    data = data[1:][::step]
    names = [j["name"] for j in joints]
    ix = {k: i for i, k in enumerate(names)}
    leg = float(np.linalg.norm(joints[ix["LeftLeg"]]["offset"]) +
                np.linalg.norm(joints[ix["LeftFoot"]]["offset"]))
    print(f"\n{path}: frames_used={len(data)} thigh+shin_off={leg:.2f}")
    print(f"  raw hips: x[{data[:,0].min():.2f},{data[:,0].max():.2f}] "
          f"y[{data[:,1].min():.2f},{data[:,1].max():.2f}] z[{data[:,2].min():.2f},{data[:,2].max():.2f}]")
    best = None
    for order in itertools.permutations("XYZ"):
        pos, _ = fk_order(joints, data, "".join(order))
        h = pos[:, ix["Hips"]]
        toe = pos[:, ix["LeftToeBase"]]
        for axis in range(3):
            sep = np.abs(h[:, axis] - toe[:, axis])
            ratio = sep.max() / leg
            tot = np.ptp(pos[:, :, axis], axis=0)
            body_h = (pos[:, ix["Head"], axis] - toe[:, axis]).max()
            line = (f"  order {''.join(order)} up-axis={axis}: max|hips_y-toe_y|/leg={ratio:5.2f} "
                    f" head-toe/leg={body_h/leg:5.2f} hip_y_med={np.median(h[:,axis])/leg:5.2f} "
                    f"travel={np.ptp(h[:,[a for a in range(3) if a!=axis][0]]):8.1f}")
            if best is None or abs(ratio - 1.0) < abs(best[0] - 1.0):
                best = (ratio, line)
            print(line)
    print(f"  BEST (ratio closest to 1.0): {best[1]}  ratio={best[0]:.2f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        score(p)
