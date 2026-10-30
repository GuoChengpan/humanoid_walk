"""量化 humanoid.xml 两处左右不对称（左膝 stiffness、左髋y armature）的实际物理影响量级。
回答问题：它们是不是'左腿被动、右腿驱动'行为不对称的主因？
"""
import numpy as np
import gymnasium as gym
import mujoco

env = gym.make("Humanoid-v4")
u = env.unwrapped
m, d = u.model, u.data
jn = [m.joint(i).name for i in range(m.njnt)]
jid = {n: i for i, n in enumerate(jn)}
jqa = {n: int(m.jnt_qposadr[i]) for i, n in enumerate(jn)}
jda = {n: int(m.jnt_dofadr[i]) for i, n in enumerate(jn)}
an = [m.actuator(i).name for i in range(m.nu)]
aid = {n: i for i, n in enumerate(an)}

print('== 参数实测（stock XML）==')
for n in ['left_knee', 'right_knee', 'left_hip_y', 'right_hip_y']:
    print(f'  {n:12s} stiffness={m.jnt_stiffness[jid[n]]:.2f} N·m/rad  '
          f'armature={m.dof_armature[jda[n]]:.4f}  '
          f'峰值力矩={m.actuator_gear[aid[n]][0]*0.4:.0f} N·m (gear{m.actuator_gear[aid[n]][0]:.0f}×0.4)')


def set_pose(knee, hip_y):
    """双膝对称弯曲、双髋对称前摆的姿态，v=0, ctrl=0"""
    q = m.qpos0.copy()
    for side in ['left', 'right']:
        q[jqa[f'{side}_knee']] = knee
        q[jqa[f'{side}_hip_y']] = hip_y
    d.qpos[:] = q
    d.qvel[:] = 0
    d.ctrl[:] = 0
    d.qacc_warmstart[:] = 0
    mujoco.mj_forward(m, d)


print('\n== 左膝弹簧力矩 vs 同姿态重力力矩（v=0, ctrl=0, 双膝同角）==')
print('   （右膝 stiffness=0，弹簧力矩恒为 0——不对称只作用在左膝）')
for knee in [-0.5, -1.0, -1.4, -2.0, -2.5]:
    set_pose(knee, -0.5)
    row = []
    for side in ['left', 'right']:
        dof = jda[f'{side}_knee']
        spring = d.qfrc_passive[dof]
        grav = d.qfrc_bias[dof]
        row.append(f'{side}: spring={spring:+6.2f} grav={grav:+7.2f} '
                   f'占比={abs(spring)/max(abs(grav), 1e-9)*100:5.1f}%')
    print(f'  knee={knee:+.1f} rad | ' + ' | '.join(row))

print('\n== 左髋y armature 对有效转动惯量的影响（qM 对角线，膝-1.0/髋-0.5 姿态）==')
set_pose(-1.0, -0.5)
Mfull = np.zeros((m.nv, m.nv))
mujoco.mj_fullM(m, Mfull, d.qM)
iL, iR = jda['left_hip_y'], jda['right_hip_y']
mL, mR = Mfull[iL, iL], Mfull[iR, iR]
print(f'  修补前: left_hip_y M_diag={mL:.4f}  right_hip_y M_diag={mR:.4f}')
m.dof_armature[iL] = 0.008                       # 修补为与右侧一致
mujoco.mj_forward(m, d)
Mfull2 = np.zeros((m.nv, m.nv))
mujoco.mj_fullM(m, Mfull2, d.qM)
mL2 = Mfull2[iL, iL]
print(f'  修补后: left_hip_y M_diag={mL2:.4f}')
print(f'  armature 差异 0.002 占左髋y有效惯量 {(0.01-0.008)/mL*100:.2f}%')
env.close()
