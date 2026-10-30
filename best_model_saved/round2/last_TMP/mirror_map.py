"""Humanoid-v4 镜像映射（跨矢状面 y→−y）+ 环境左右对称修补。

符号规则均由 debug_probes/build_mirror_map.py 四层验证：
  V1 往返 ✓ | qacc 等变性（修补后）4e-12 ✓ | 零动作整段重放 1e-17（机器精度）✓ |
  qpos0 body 位姿/质量/gear 体检 ✓

规则来源（机制级）：
  - 四元数镜像 = 共轭 R' = M·R·M⁻¹（M=diag(1,-1,1)）⇒ (w,x,y,z)→(w,−x,+y,−z)
  - 根角速度是轴矢量：全局/局部帧下符号规则相同 (wx,wy,wz)→(−wx,+wy,−wz)
  - 关节/动作符号 = 轴规则 pair_sign：伙伴轴与 M·自轴同向→角度变号，反向→不变号
    （左腿 hip_x/hip_z 轴在 XML 中取反 ⇒ 符号 +1；肩1/肩2 ⇒ −1；肘 ⇒ +1；
      abdomen_z/x 自映射变号 −1，abdomen_y 不变号 +1）
"""
import numpy as np

M_REF = np.diag([1.0, -1.0, 1.0])   # 矢状面镜像（y→−y）

PAIRS = [('right_hip_x', 'left_hip_x'), ('right_hip_z', 'left_hip_z'),
         ('right_hip_y', 'left_hip_y'), ('right_knee', 'left_knee'),
         ('right_shoulder1', 'left_shoulder1'), ('right_shoulder2', 'left_shoulder2'),
         ('right_elbow', 'left_elbow')]


def patch_mirror_symmetry(model):
    """修补 stock humanoid.xml 的两处左右不对称（gymnasium 上游历史怪癖，非本项目引入）：
      left_knee  jnt_stiffness = 1.0  而 right_knee = 0 —— 左膝被弹簧恒拉直
      left_hip_y dof_armature  = 0.01 而 right_hip_y = 0.008 —— 左髋转子惯量高 25%
    取右腿值统一。修补后动力学镜像等变性实测 4e-12（build_mirror_map.py §6.8）。
    不改 obs / qpos0 / reset，对 warm-start 模型透明。"""
    jn = [model.joint(i).name for i in range(model.njnt)]
    ji = {n: i for i, n in enumerate(jn)}
    jd = {n: int(model.jnt_dofadr[i]) for i, n in enumerate(jn)}
    model.jnt_stiffness[ji['left_knee']] = 0.0
    model.dof_armature[jd['left_hip_y']] = 0.008


def _pair_sign(axis_self, axis_partner):
    """镜像共轭 R(M·a,−θ)：伙伴轴与 M·a 同向→变号(−1)，反向→不变号(+1)"""
    return -int(np.sign(np.dot(axis_partner, M_REF @ axis_self)))


class MirrorMap:
    """由 live model 构建镜像映射：mirror_qpos_qvel / mirror_action"""

    def __init__(self, model):
        jnames = [model.joint(i).name for i in range(model.njnt)]
        anames = [model.actuator(i).name for i in range(model.nu)]
        jaxis = {n: np.array(model.jnt_axis[i]) for i, n in enumerate(jnames)}
        self._jqadr = {n: int(model.jnt_qposadr[i]) for i, n in enumerate(jnames)}
        self._jdadr = {n: int(model.jnt_dofadr[i]) for i, n in enumerate(jnames)}
        self._jn = jnames

        # --- 关节镜像表 ---
        self.jmirror = {}
        for r, l in PAIRS:
            self.jmirror[r] = (l, _pair_sign(jaxis[r], jaxis[l]))
            self.jmirror[l] = (r, _pair_sign(jaxis[l], jaxis[r]))
        for n in ['abdomen_z', 'abdomen_y', 'abdomen_x']:
            # 自映射：sign=−sign(a·(M·a))。传 M@a 会恒得 −1，abdomen_y 会被错翻
            self.jmirror[n] = (n, _pair_sign(jaxis[n], jaxis[n]))

        # --- 动作镜像表（17，按执行器序）---
        aid = {n: i for i, n in enumerate(anames)}
        self.act_src = np.arange(len(anames))
        self.act_sign = np.ones(len(anames), dtype=np.float64)
        for i, a in enumerate(anames):
            if a.startswith('right_'):
                partner, sign = 'left_' + a[6:], _pair_sign(jaxis[a], jaxis['left_' + a[6:]])
            elif a.startswith('left_'):
                partner, sign = 'right_' + a[5:], _pair_sign(jaxis[a], jaxis['right_' + a[5:]])
            elif 'abdomen' in a:
                partner, sign = a, _pair_sign(jaxis[a], jaxis[a])
            else:
                raise ValueError(f'未覆盖的执行器: {a}')
            self.act_src[i] = aid[partner]
            self.act_sign[i] = sign

        # --- obs 镜像表（376）---
        # 块结构: [qpos去xy 0:22][qvel 22:45][cinert 45:185][cvel 185:269][qfrc 269:292][cfrc 292:376]
        # 与 debug_probes/build_mirror_map.py 验证版一致（V1 往返 ✓ + 端到端 obs 对比 ✓）
        bnames = [model.body(i).name for i in range(model.nbody)]
        bid = {n: i for i, n in enumerate(bnames)}
        bmirror = {}
        for b in bnames:
            if b.startswith('right_'):
                bmirror[b] = 'left_' + b[6:]
            elif b.startswith('left_'):
                bmirror[b] = 'right_' + b[5:]
            else:
                bmirror[b] = b
        self.obs_src = np.arange(376)
        self.obs_sign = np.ones(376)
        qpos_map = {0: (2, +1), 1: (3, +1), 2: (4, -1), 3: (5, +1), 4: (6, -1)}
        for k, jn in enumerate(jnames[1:]):
            p, s = self.jmirror[jn]
            qpos_map[5 + k] = (self._jqadr[p], s)
        for oi, (qi, sg) in qpos_map.items():
            self.obs_src[oi] = qi - 2          # obs qpos 块 = qpos[2:24]
            self.obs_sign[oi] = sg
        qvel_map = {22: (22, +1), 23: (23, -1), 24: (24, +1),
                    25: (25, -1), 26: (26, +1), 27: (27, -1)}
        for k, jn in enumerate(jnames[1:]):
            p, s = self.jmirror[jn]
            qvel_map[28 + k] = (22 + self._jdadr[p], s)
        for oi, (vi, sg) in qvel_map.items():
            self.obs_src[oi] = vi
            self.obs_sign[oi] = sg
        cinert_sign10 = np.array([+1, +1, +1, -1, +1, -1, +1, -1, +1, +1])
        for bi, bn in enumerate(bnames):
            pi = bid[bmirror[bn]]
            for v in range(10):
                self.obs_src[45 + bi * 10 + v] = 45 + pi * 10 + v
                self.obs_sign[45 + bi * 10 + v] = cinert_sign10[v]
        cvel_sign6 = np.array([-1, +1, -1, +1, -1, +1])
        for bi, bn in enumerate(bnames):
            pi = bid[bmirror[bn]]
            for v in range(6):
                self.obs_src[185 + bi * 6 + v] = 185 + pi * 6 + v
                self.obs_sign[185 + bi * 6 + v] = cvel_sign6[v]
        for oi, (vi, sg) in qvel_map.items():
            self.obs_src[269 + (oi - 22)] = vi + 247      # qfrc 块：obs 索引平移 247
            self.obs_sign[269 + (oi - 22)] = sg
        cfrc_sign6 = np.array([+1, -1, +1, -1, +1, -1])
        for bi, bn in enumerate(bnames):
            pi = bid[bmirror[bn]]
            for v in range(6):
                self.obs_src[292 + bi * 6 + v] = 292 + pi * 6 + v
                self.obs_sign[292 + bi * 6 + v] = cfrc_sign6[v]

    def mirror_action(self, a):
        return np.asarray(a)[self.act_src] * self.act_sign

    def mirror_obs(self, x):
        """376 维观测镜像（x[obs_src]*obs_sign）。往返对合已由 build_mirror_map.py V1 验证。"""
        return np.asarray(x)[self.obs_src] * self.obs_sign

    def mirror_qpos_qvel(self, qpos, qvel):
        """完整物理状态镜像（set_state 用）。从原状态读、写入副本，L↔R 互换。"""
        q = np.array(qpos, dtype=np.float64)
        v = np.array(qvel, dtype=np.float64)
        q[1] = -q[1]                                   # 根 y 取反
        q[4] = -q[4]; q[6] = -q[6]                     # 四元数 (w,x,y,z)→(w,−x,+y,−z)
        v[1] = -v[1]                                   # 根 vy
        v[3] = -v[3]; v[5] = -v[5]                     # 根 wx、wz 变号（wy 不变）
        for jn in self._jn[1:]:
            p, sgn = self.jmirror[jn]
            q[self._jqadr[p]] = sgn * qpos[self._jqadr[jn]]
            v[self._jdadr[p]] = sgn * qvel[self._jdadr[jn]]
        return q, v
