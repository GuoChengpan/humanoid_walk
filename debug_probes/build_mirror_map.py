"""镜像映射构建 + 三重验证（阶段 1.2-1.4）
生成 376 维 obs / 17 维 action 的镜像映射，验证：
  V1 往返: mirror(mirror(x)) == x
  V2 物理: 镜像态 set_state 后 step 不炸
  V3 轨迹: 镜像初始态 + 镜像动作序列 的轨迹 = 原轨迹的逐帧镜像（金标准）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import gymnasium as gym

env = gym.make("Humanoid-v4", max_episode_steps=1500)
m = env.unwrapped.model
d = env.unwrapped.data

# ========== 1. 结构枚举 ==========
nq, nv, nbody, nu = m.nq, m.nv, m.nbody, m.nu
jnames = [m.joint(i).name for i in range(m.njnt)]
bnames = [m.body(i).name for i in range(m.nbody)]
anames = [m.actuator(i).name for i in range(m.nu)]
jaxis = {n: np.array(m.jnt_axis[i]) for i, n in enumerate(jnames)}
jqadr = {n: int(m.jnt_qposadr[i]) for i, n in enumerate(jnames)}
jdadr = {n: int(m.jnt_dofadr[i]) for i, n in enumerate(jnames)}

print(f'nq={nq} nv={nv} nbody={nbody} nu={nu} | obs=376')
print('关节轴:')
for n in jnames[1:]:
    a = jaxis[n]
    tag = 'y轴(不变号)' if abs(a[1]) > 0.9 else '面内轴(变号)'
    print(f'  {n:18s} axis=({a[0]:+.0f},{a[1]:+.0f},{a[2]:+.0f}) {tag}')

# ========== 2. 关节镜像对 + 符号（轴规则自动生成）==========
# L↔R 对
pairs = [('right_hip_x','left_hip_x'), ('right_hip_z','left_hip_z'),
         ('right_hip_y','left_hip_y'), ('right_knee','left_knee'),
         ('right_shoulder1','left_shoulder1'), ('right_shoulder2','left_shoulder2'),
         ('right_elbow','left_elbow')]
M_ref = np.diag([1.0, -1.0, 1.0])   # 矢状面镜像（y→-y）

def pair_sign(axis_self, axis_partner):
    """镜像共轭 R(M·a,-θ)：伙伴轴与 M·a 同向→变号，反向→不变号"""
    return -int(np.sign(np.dot(axis_partner, M_ref @ axis_self)))

jmirror = {}   # name -> (partner_name, sign)
for r, l in pairs:
    jmirror[r] = (l, pair_sign(jaxis[r], jaxis[l]))
    jmirror[l] = (r, pair_sign(jaxis[l], jaxis[r]))
# 腹部（自映射）：共轭 R(a,θ)→R(M·a,−θ)，同轴时 sign=−sign(a·(M·a))
# 面内轴(z/x)变号、y 轴不变号；传 M@a 会恒得 −1，abdomen_y 被错翻
for n in ['abdomen_z', 'abdomen_y', 'abdomen_x']:
    jmirror[n] = (n, pair_sign(jaxis[n], jaxis[n]))

print('\n关节镜像规则:')
for n, (p, s) in jmirror.items():
    print(f'  {n:18s} -> {p:18s} sign={s:+d}')

# ========== 3. body 镜像对（cinert/cvel/cfrc 块用）==========
bmirror = {}
for b in bnames:
    if b.startswith('right_'):
        bmirror[b] = 'left_' + b[6:]
    elif b.startswith('left_'):
        bmirror[b] = 'right_' + b[5:]
    else:
        bmirror[b] = b
bid = {n: i for i, n in enumerate(bnames)}

# ========== 4. obs 映射表（376）==========
# 块结构: [qpos去xy: 0:22] [qvel: 22:45] [cinert: 45:185] [cvel: 185:269] [qfrc: 269:292] [cfrc: 292:376]
B = {'qpos': (0, 22), 'qvel': (22, 45), 'cinert': (45, 185), 'cvel': (185, 269),
     'qfrc': (269, 292), 'cfrc': (292, 376)}

obs_src = np.arange(376)
obs_sign = np.ones(376)

# --- qpos 块 (obs[0:22] ← qpos[2:24]) ---
# obs[0]=z, obs[1:5]=quat(w,x,y,z), obs[5+i]=jnt[i+1] 的 qpos
# 镜像: z 不变; quat (w,x,y,z)->(w,x,-y,-z); 关节按 jmirror
# qpos 布局: [x=0, y=1, z=2, qw=3, qx=4, qy=5, qz=6, joints 7..23]
# obs 索引 → qpos 索引 + 符号
qpos_map = {0: (2, +1),                     # z
            1: (3, +1), 2: (4, -1),         # qw, qx(变号)
            3: (5, +1), 4: (6, -1)}         # qy(不变), qz(变号)：镜像 (w,x,y,z)→(w,−x,+y,−z)
jn_obs_order = jnames[1:]                    # 17 关节按 XML 序
for k, jn in enumerate(jn_obs_order):
    partner, sign = jmirror[jn]
    qpos_map[5 + k] = (jqadr[partner], sign)
for oi, (qi, sg) in qpos_map.items():
    obs_src[oi] = qi - 2                     # obs 的 qpos 块 = qpos[2:24]：obs 索引 = qpos 地址 - 2
    obs_sign[oi] = sg

# --- qvel 块 (obs[22:45] ← qvel[0:23]) ---
# 根 6 dof: [vx, vy, vz, wx, wy, wz]（自由度全局/局部帧由轨迹验证裁定）
qvel_map = {22: (22, +1), 23: (23, -1), 24: (24, +1),    # vx, vy(变号), vz（值=obs 块内索引）
            25: (25, -1), 26: (26, +1), 27: (27, -1)}    # wx(变号), wy(不变), wz(变号) 伪矢量
for k, jn in enumerate(jn_obs_order):
    partner, sign = jmirror[jn]
    qvel_map[28 + k] = (22 + jdadr[partner], sign)
for oi, (vi, sg) in qvel_map.items():
    obs_src[oi] = vi
    obs_sign[oi] = sg

# --- cinert (obs[45:185] = 14 body × 10) ---
# 布局假设: [Ixx,Iyy,Izz,Ixy,Ixz,Iyz, cx,cy,cz, mass]
# 镜像: 含 y 的惯性积(Ixy,Iyz)变号、com_y 变号、其余不变；body 左右互换
cinert_sign10 = np.array([+1,+1,+1,-1,+1,-1, +1,-1,+1, +1])
for bi, bn in enumerate(bnames):
    partner = bmirror[bn]
    pi = bid[partner]
    for v in range(10):
        oi = 45 + bi*10 + v
        obs_src[oi] = 45 + pi*10 + v
        obs_sign[oi] = cinert_sign10[v]

# --- cvel (obs[185:269] = 14 body × 6) ---
# 布局: [angular(3), linear(3)]；angular=axial(ωx,ωz 变号), linear=polar(vy 变号)
cvel_sign6 = np.array([-1,+1,-1, +1,-1,+1])
for bi, bn in enumerate(bnames):
    partner = bmirror[bn]
    pi = bid[partner]
    for v in range(6):
        oi = 185 + bi*6 + v
        obs_src[oi] = 185 + pi*6 + v
        obs_sign[oi] = cvel_sign6[v]

# --- qfrc (obs[269:292] = 23) ---
# 与 qvel 同规则（根 6 dof 力=0 恒定 + 17 关节）
# qfrc 块（obs[269:292]）：同 qvel 规则，obs 索引平移 247（=269−22）
for oi, (vi, sg) in qvel_map.items():
    obs_src[269 + (oi-22)] = vi + 247
    obs_sign[269 + (oi-22)] = sg

# --- cfrc (obs[292:376] = 14 body × 6) ---
# 布局: [force(3) polar, torque(3) axial]
cfrc_sign6 = np.array([+1,-1,+1, -1,+1,-1])
for bi, bn in enumerate(bnames):
    partner = bmirror[bn]
    pi = bid[partner]
    for v in range(6):
        oi = 292 + bi*6 + v
        obs_src[oi] = 292 + pi*6 + v
        obs_sign[oi] = cfrc_sign6[v]

# ========== 5. 动作映射 (17) ==========
act_src = np.arange(17)
act_sign = np.ones(17)
amirror = {}
for a in anames:
    if a.startswith('right_'):
        amirror[a] = ('left_' + a[6:], pair_sign(jaxis[a], jaxis['left_' + a[6:]]))
    elif a.startswith('left_'):
        amirror[a] = ('right_' + a[5:], pair_sign(jaxis[a], jaxis['right_' + a[5:]]))
    elif 'abdomen' in a:
        amirror[a] = (a, pair_sign(jaxis[a], jaxis[a]))
aid = {n: i for i, n in enumerate(anames)}
for i, a in enumerate(anames):
    partner, sign = amirror[a]
    act_src[i] = aid[partner]
    act_sign[i] = sign

# ========== 6. 镜像函数 ==========
def mirror_obs(x):
    return x[obs_src] * obs_sign

def mirror_action(a):
    return a[act_src] * act_sign

def mirror_qpos_qvel(qpos, qvel):
    """完整环境状态镜像（用于 set_state 轨迹测试）"""
    q = qpos.copy()
    q[1] = -q[1]                                   # y 取反
    # 四元数镜像（跨 y=0 平面共轭 R'=MRM⁻¹）: (w,x,y,z)→(w,−x,+y,−z)
    q[4] = -q[4]; q[6] = -q[6]
    for jn in jnames[1:]:
        p, sgn = jmirror[jn]
        q[jqadr[p]] = sgn * qpos[jqadr[jn]]
    v = qvel.copy()
    v[1] = -v[1]                                   # vy
    v[3] = -v[3]; v[5] = -v[5]                     # wx, wz（伪矢量，全局帧假设）
    for jn in jnames[1:]:
        p, sgn = jmirror[jn]
        v[jdadr[p]] = sgn * qvel[jdadr[jn]]
    return q, v

def mirror_vec23(u):
    """nv 维向量（qvel/qacc 类）镜像：根 (−,+,−) + 关节规则，用于 qacc_warmstart"""
    w = u.copy()
    w[1] = -u[1]; w[3] = -u[3]; w[5] = -u[5]
    for jn in jnames[1:]:
        p, sgn = jmirror[jn]
        w[jdadr[p]] = sgn * u[jdadr[jn]]
    return w

QN = ['x','y','z','qw','qx','qy','qz'] + list(jnames[1:])
QV = ['vx','vy','vz','wx','wy','wz'] + list(jnames[1:])

# ========== 6.5 零动作单步对称测试（隔离状态镜像，无策略） ==========
print('\n========== 零动作单步对称测试 ==========')
import mujoco as _mj
env3 = gym.make("Humanoid-v4", max_episode_steps=1500)
mj3 = env3.unwrapped
obs3, _ = env3.reset(seed=7)
for t in range(8):
    q0, v0 = mj3.data.qpos.copy(), mj3.data.qvel.copy()
    ws0 = mj3.data.qacc_warmstart.copy()
    obs3, *_ = env3.step(np.zeros(17))
    q1, v1 = mj3.data.qpos.copy(), mj3.data.qvel.copy()
    qm, vm = mirror_qpos_qvel(q0, v0)
    mj3.data.qpos[:] = qm; mj3.data.qvel[:] = vm
    mj3.data.qacc_warmstart[:] = mirror_vec23(ws0)
    _mj.mj_forward(mj3.model, mj3.data)
    obs3m, *_ = env3.step(np.zeros(17))
    q_exp, v_exp = mirror_qpos_qvel(q1, v1)
    eq = np.abs(mj3.data.qpos - q_exp)
    ev = np.abs(mj3.data.qvel - v_exp)
    iq, iv = int(eq.argmax()), int(ev.argmax())
    print(f'  t={t}: q err={eq.max():.2e} @{QN[iq]} | v err={ev.max():.2e} @{QV[iv]}')
    mj3.data.qpos[:] = q1; mj3.data.qvel[:] = v1
    _mj.mj_forward(mj3.model, mj3.data)
env3.close()

# ========== 6.6 状态级动力学对称测试（qacc 单次 forward + 单次 mj_step，无混沌） ==========
print('\n========== qacc / 单步 mj_step 对称测试 ==========')
env4 = gym.make("Humanoid-v4", max_episode_steps=1500)
mj4 = env4.unwrapped
rng4 = np.random.default_rng(3)
for trial in range(3):
    obs4, _ = env4.reset(seed=10 + trial)
    q4, v4 = mj4.data.qpos.copy(), mj4.data.qvel.copy()
    a4 = rng4.uniform(-0.4, 0.4, 17)
    for with_a in (False, True):
        tag = 'a≠0' if with_a else 'a=0'
        # A: 原状态 → mj_forward
        mj4.data.qpos[:] = q4; mj4.data.qvel[:] = v4
        mj4.data.qacc_warmstart[:] = 0
        mj4.data.ctrl[:] = a4 if with_a else 0.0
        _mj.mj_forward(mj4.model, mj4.data)
        qaccA = mj4.data.qacc.copy(); nconA = mj4.data.ncon
        # B: 镜像状态 + 镜像动作 → mj_forward
        qm4, vm4 = mirror_qpos_qvel(q4, v4)
        mj4.data.qpos[:] = qm4; mj4.data.qvel[:] = vm4
        mj4.data.qacc_warmstart[:] = 0
        mj4.data.ctrl[:] = mirror_action(a4) if with_a else 0.0
        _mj.mj_forward(mj4.model, mj4.data)
        qaccB = mj4.data.qacc.copy(); nconB = mj4.data.ncon
        err = np.abs(qaccA - mirror_vec23(qaccB))
        i = int(err.argmax())
        print(f'  seed={10+trial} {tag}: qacc err={err.max():.2e} @{QV[i]}  ncon A/B={nconA}/{nconB}')
        # C: 单次 mj_step（frame_skip=1，绕过 gym）
        mj4.data.qpos[:] = q4; mj4.data.qvel[:] = v4
        mj4.data.qacc_warmstart[:] = 0
        mj4.data.ctrl[:] = a4 if with_a else 0.0
        _mj.mj_step(mj4.model, mj4.data, nstep=1)
        q1c, v1c = mj4.data.qpos.copy(), mj4.data.qvel.copy()
        mj4.data.qpos[:] = qm4; mj4.data.qvel[:] = vm4
        mj4.data.qacc_warmstart[:] = 0
        mj4.data.ctrl[:] = mirror_action(a4) if with_a else 0.0
        _mj.mj_step(mj4.model, mj4.data, nstep=1)
        q_exp, v_exp = mirror_qpos_qvel(q1c, v1c)
        eqc = np.abs(mj4.data.qpos - q_exp); evc = np.abs(mj4.data.qvel - v_exp)
        print(f'           mj_step err q={eqc.max():.2e}@{QN[int(eqc.argmax())]} '
              f'v={evc.max():.2e}@{QV[int(evc.argmax())]}')
env4.close()

# ========== 6.7 模型镜像对称性体检 ==========
print('\n========== 模型对称性体检 ==========')
env5 = gym.make("Humanoid-v4", max_episode_steps=1500)
mj5 = env5.unwrapped
mm, md = mj5.model, mj5.data

n_asym = 0
for b in bnames:
    p = bmirror[b]
    if p == b: continue
    i1, i2 = bid[b], bid[p]
    dm = mm.body_mass[i1] - mm.body_mass[i2]
    di = np.abs(mm.body_inertia[i1] - mm.body_inertia[i2]).max()
    if abs(dm) > 1e-12 or di > 1e-12:
        print(f'  ✗ 质量/惯量: {b} vs {p}: Δm={dm:.3e} ΔI={di:.3e}')
        n_asym += 1
print(f'  5.1 质量惯量: {"✓ 对称" if n_asym == 0 else f"✗ {n_asym} 对不对称"}')

md.qpos[:] = mm.qpos0; md.qvel[:] = 0.0
md.ctrl[:] = 0.0; md.qacc_warmstart[:] = 0.0
_mj.mj_forward(mm, md)
n_bad = 0
for b in bnames:
    p = bmirror[b]; i1, i2 = bid[b], bid[p]
    qL, qR = md.xquat[i1], md.xquat[i2]
    if p != b:
        qr = np.array([qR[0], -qR[1], qR[2], -qR[3]])
        if not (np.allclose(qL, qr, atol=1e-10) or np.allclose(qL, -qr, atol=1e-10)):
            print(f'  ✗ xquat 非镜像: {b}{np.round(qL,4)} vs M·{p}{np.round(qr,4)}')
            n_bad += 1
        pL, pR = md.xipos[i1], md.xipos[i2]
        if not np.allclose(pL, pR * np.array([1, -1, 1]), atol=1e-10):
            print(f'  ✗ xipos 非镜像: {b}{np.round(pL,4)} vs M·{p}{np.round(pR*[1,-1,1],4)}')
            n_bad += 1
    else:
        if abs(qL[1]) > 1e-10 or abs(qL[3]) > 1e-10:
            print(f'  ✗ 自镜像 body {b} quat={np.round(qL,4)} 含 x/z 旋转')
            n_bad += 1
        if abs(md.xipos[i1][1]) > 1e-10:
            print(f'  ✗ 自镜像 body {b} 质心 y={md.xipos[i1][1]:.5f}≠0')
            n_bad += 1
print(f'  5.2 qpos0 body 位姿镜像: {"✓" if n_bad == 0 else f"✗ {n_bad} 项"}')

print('  5.3 关节世界轴规则 (qpos0):')
for r, l in pairs:
    ar, al = md.xaxis[jnames.index(r)], md.xaxis[jnames.index(l)]
    if np.allclose(al, M_ref @ ar, atol=1e-9): sw = -1
    elif np.allclose(al, -(M_ref @ ar), atol=1e-9): sw = +1
    else: sw = 0
    sl = jmirror[r][1]
    mark = '' if sw == sl else '   ← 与局部轴规则不一致!'
    print(f'    {r:18s} aR={np.round(ar,3)} aL={np.round(al,3)} 局部={sl:+d} 世界={sw:+d}{mark}')
for n in ['abdomen_z', 'abdomen_y', 'abdomen_x']:
    a = md.xaxis[jnames.index(n)]
    if np.allclose(a, M_ref @ a, atol=1e-9): sw = -1
    elif np.allclose(a, -(M_ref @ a), atol=1e-9): sw = +1
    else: sw = 0
    sl = jmirror[n][1]
    mark = '' if sw == sl else '   ← 不一致!'
    print(f'    {n:18s} a={np.round(a,3)} 局部={sl:+d} 世界={sw:+d}{mark}')

n_gear = 0
for a in anames:
    if a.startswith(('right_', 'left_')):
        partner = ('left_' + a[6:]) if a.startswith('right_') else ('right_' + a[5:])
        g1, g2 = mm.actuator_gear[aid[a]][0], mm.actuator_gear[aid[partner]][0]
        if abs(g1 - g2) > 1e-12:
            print(f'  ✗ gear 不对称: {a}={g1} vs {partner}={g2}')
            n_gear += 1
print(f'  5.4 actuator gear: {"✓ 对称" if n_gear == 0 else "✗ 有不对称"}')

obs5, _ = env5.reset(seed=10)
q5, v5 = mj5.data.qpos.copy(), mj5.data.qvel.copy()
a5 = rng4.uniform(-0.4, 0.4, 17)
for with_a in (False, True):
    mj5.data.qpos[:] = q5; mj5.data.qvel[:] = v5
    mj5.data.qacc_warmstart[:] = 0
    mj5.data.ctrl[:] = a5 if with_a else 0.0
    _mj.mj_forward(mm, md)
    qa = md.qacc.copy()
    qm5, vm5 = mirror_qpos_qvel(q5, v5)
    mj5.data.qpos[:] = qm5; mj5.data.qvel[:] = vm5
    mj5.data.qacc_warmstart[:] = 0
    mj5.data.ctrl[:] = mirror_action(a5) if with_a else 0.0
    _mj.mj_forward(mm, md)
    err = np.abs(qa - mirror_vec23(md.qacc))
    top = np.argsort(err)[::-1][:6]
    tag = 'a≠0' if with_a else 'a=0'
    print(f'  5.5 qacc err {tag}: ' + ', '.join(f'{QV[i]}={err[i]:.2e}' for i in top))
env5.close()

# ========== 6.8 非对称修补验证 ==========
print('\n========== 修补验证 (left_knee stiffness→0, left_hip_y armature→0.008) ==========')
env6 = gym.make("Humanoid-v4", max_episode_steps=1500)
mj6 = env6.unwrapped
mm6, md6 = mj6.model, mj6.data
jd6 = {n: int(mm6.jnt_dofadr[i]) for i, n in enumerate([mm6.joint(i).name for i in range(mm6.njnt)])}
ji6 = {n: i for i, n in enumerate([mm6.joint(i).name for i in range(mm6.njnt)])}
print(f'  原值: LK stiff={mm6.jnt_stiffness[ji6["left_knee"]]} RK stiff={mm6.jnt_stiffness[ji6["right_knee"]]} | '
      f'LHy arm={mm6.dof_armature[jd6["left_hip_y"]]} RHy arm={mm6.dof_armature[jd6["right_hip_y"]]}')
mm6.jnt_stiffness[ji6['left_knee']] = 0.0
mm6.dof_armature[jd6['left_hip_y']] = 0.008
obs6, _ = env6.reset(seed=10)
q6, v6 = mj6.data.qpos.copy(), mj6.data.qvel.copy()
a6 = rng4.uniform(-0.4, 0.4, 17)
for with_a in (False, True):
    mj6.data.qpos[:] = q6; mj6.data.qvel[:] = v6
    mj6.data.qacc_warmstart[:] = 0
    mj6.data.ctrl[:] = a6 if with_a else 0.0
    _mj.mj_forward(mm6, md6)
    qa = md6.qacc.copy()
    qm6, vm6 = mirror_qpos_qvel(q6, v6)
    mj6.data.qpos[:] = qm6; mj6.data.qvel[:] = vm6
    mj6.data.qacc_warmstart[:] = 0
    mj6.data.ctrl[:] = mirror_action(a6) if with_a else 0.0
    _mj.mj_forward(mm6, md6)
    err = np.abs(qa - mirror_vec23(md6.qacc))
    print(f'  修补后 qacc err {"a≠0" if with_a else "a=0"}: {err.max():.2e} @{QV[int(err.argmax())]}')
obs6, _ = env6.reset(seed=7)
for t in range(8):
    q0b, v0b = mj6.data.qpos.copy(), mj6.data.qvel.copy()
    ws0 = mj6.data.qacc_warmstart.copy()
    obs6, *_ = env6.step(np.zeros(17))
    q1b, v1b = mj6.data.qpos.copy(), mj6.data.qvel.copy()
    qm, vm = mirror_qpos_qvel(q0b, v0b)
    mj6.data.qpos[:] = qm; mj6.data.qvel[:] = vm
    mj6.data.qacc_warmstart[:] = mirror_vec23(ws0)
    _mj.mj_forward(mm6, md6)
    obs6, *_ = env6.step(np.zeros(17))
    q_exp, v_exp = mirror_qpos_qvel(q1b, v1b)
    eq = np.abs(mj6.data.qpos - q_exp); ev = np.abs(mj6.data.qvel - v_exp)
    print(f'  修补后零动作 t={t}: q err={eq.max():.2e} | v err={ev.max():.2e}')
    mj6.data.qpos[:] = q1b; mj6.data.qvel[:] = v1b
    _mj.mj_forward(mm6, md6)
env6.close()

# ========== 7. 验证 ==========
print('\n========== V1 往返验证 =========')
rng = np.random.default_rng(0)
x = rng.standard_normal(376)
ok1 = np.allclose(mirror_obs(mirror_obs(x)), x)
ok2 = np.allclose(obs_src[obs_src], np.arange(376))
ok3 = np.allclose(obs_sign[obs_src] * obs_sign, np.ones(376))
print(f'往返 mirror²==id: {ok1} | src[src]==id: {ok2} | sign 对合: {ok3}')
a = rng.standard_normal(17)
oka = np.allclose(mirror_action(mirror_action(a)), a)
print(f'动作往返: {oka}')
V1 = ok1 and ok2 and ok3 and oka

print('\n========== V2+V3 物理/轨迹镜像验证（金标准）==========')
# 加载 560K 模型走一段，取原轨迹；镜像初始态+镜像动作重放，对比
from stable_baselines3 import SAC
import train_humanoid_sac as T

model = SAC.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "best_model", "best_model.zip"))
env2 = gym.make("Humanoid-v4", max_episode_steps=1500)
env2 = T.WalkRewardWrapper(env2)
# 运行时对称修补（与 6.8 相同）：左膝 stiffness 1→0、左髋y armature 0.01→0.008
_mj2 = env2.unwrapped
_jd = {n: int(_mj2.model.jnt_dofadr[i]) for i, n in enumerate(
    [_mj2.model.joint(i).name for i in range(_mj2.model.njnt)])}
_ji = {n: i for i, n in enumerate([_mj2.model.joint(i).name for i in range(_mj2.model.njnt)])}
_mj2.model.jnt_stiffness[_ji['left_knee']] = 0.0
_mj2.model.dof_armature[_jd['left_hip_y']] = 0.008
print('[V2/V3 env 已修补: left_knee stiffness=0, left_hip_y armature=0.008]')

mj = env2.unwrapped
obs, _ = env2.reset(seed=52)
# 收集原轨迹（obs+状态+动作）
states, acts, obs_hist = [], [], []
done = False; n = 0
while not done and n < 40:
    a, _ = model.predict(obs, deterministic=True)
    obs_hist.append(obs.copy())
    states.append((mj.data.qpos.copy(), mj.data.qvel.copy()))
    acts.append(a.copy())
    obs, r, term, trunc, info = env2.step(a)
    n += 1
    done = term or trunc
print(f'原轨迹 {len(states)} 步')

# 镜像初始态重放
q0, v0 = states[0]
qm, vm = mirror_qpos_qvel(q0, v0)
mj.data.qpos[:] = qm
mj.data.qvel[:] = vm
import mujoco
mujoco.mj_forward(mj.model, mj.data)
ok_traj = True
max_err = 0.0
max_err_obs = 0.0
for i in range(len(states) - 1):
    am = mirror_action(acts[i])
    obs_m, r_m, term_m, trunc_m, info_m = env2.step(am)
    q_now, v_now = mj.data.qpos.copy(), mj.data.qvel.copy()
    q_exp, v_exp = mirror_qpos_qvel(*states[i+1])
    err_q = np.abs(q_now - q_exp).max()
    err_v = np.abs(v_now - v_exp).max()
    max_err = max(max_err, err_q, err_v)
    # 端到端：镜像态的实际 obs == mirror_obs(原 obs)，校验 cinert/cvel/qfrc/cfrc 全部符号映射
    err_obs = np.abs(obs_m - mirror_obs(obs_hist[i+1])).max()
    max_err_obs = max(max_err_obs, err_obs)
    if err_q > 1e-4 or err_v > 1e-3 or err_obs > 1e-2:
        ok_traj = False
        dq = np.abs(q_now - q_exp); dv = np.abs(v_now - v_exp)
        print(f'  步 {i}: 偏差 q={err_q:.4f}@{QN[int(dq.argmax())]} '
              f'v={err_v:.4f}@{QV[int(dv.argmax())]} obs={err_obs:.4f} ← 镜像不匹配')
        if i > 10: break
print(f'轨迹镜像重放: {"✓ 通过" if ok_traj else "✗ 失败"} 最大偏差 q/v={max_err:.4f} obs={max_err_obs:.4f}')
print(f'\n========== 总结 ==========')
print(f'V1 往返: {"✓" if V1 else "✗"} | V2/V3 轨迹镜像: {"✓" if ok_traj else "✗"}')
