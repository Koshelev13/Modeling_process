import pybullet as p
import time
import math
import numpy as np

GUI = False
OWN_REG = True

# Параметры системы
g = 10 # Ускорение свободного падения
m = 1 # Масса маятника
L = 0.5  # Длина маятника
kf = 0.1  # Коэффициент трения

q0 = 0 # Начальное положение
dt = 1 / 240 # Временной шаг
t = 0 # Текущее время
maxTime = 5 # Максимальное время моделирования
qd = math.pi / 2 # Желаемое положение
trajTime = 4 # Время траектории

jointIdx = 1 # joint index

if (GUI):
    physicsClient = p.connect(p.GUI)
else:
    physicsClient = p.connect(p.DIRECT)

p.setGravity(0, 0, -g)

bodyId = p.loadURDF("./pendulum.urdf")

# Отключение затухания сочленения
p.changeDynamics(bodyId, jointIdx, linearDamping = 0)

# Переход в начальную позицию
p.setJointMotorControl2(bodyIndex=bodyId,
                        jointIndex=jointIdx,
                        targetPosition=q0,
                        controlMode=p.POSITION_CONTROL)

for _ in range(1000):
    p.stepSimulation()

q0_fact = p.getJointState(bodyId, jointIdx)[0]
print(f'q0 fact: {q0_fact}') # Фактическая начальная позиция
print(f'q0 error: {q0 - q0_fact}') # Ошибка начальной позиции

kp = 1600
kv = 120

"""
    pos: текущая позиция маятника
    vel: текущая скорость маятника
    pos_d: желаемая позиция маятника
    vel_d: желаемая скорость маятника
    ff_acc: предварительное ускорение

    Возвращает:
    управляющее воздействие для маятника
"""
def feedback_lin(pos, vel, pos_d, vel_d, ff_acc):
    u_nonlin = (g / L) * math.sin(pos) + kf / (m * L * L) * vel
    u_lin = ff_acc - kp * (pos - pos_d) - kv * (vel - vel_d)
    # Функция вычисляет управляющее воздействие для маятника. Линейное и нелинейное
    ctrl = m * L * L * (u_nonlin + u_lin)
    return ctrl

def cubic_interpol(pos_0, pos_d, T, t):
    delta = pos_d - pos_0
    # a0 = 0
    # a1 = 0
    a2 = 3 / T ** 2
    a3 = -2 / T ** 3
    s = a3 * t ** 3 + a2 * t ** 2

    return ((pos_0 + s * delta), delta * (2 * a2 * t + 3 * a3 * t ** 2),
            delta * (2 * a2 + 6 * a3 * t)) if (t <= T) else (pos_d, 0, 0)

"""
    Функция fifth_interpol вычисляет параметры пятого порядка траектории между начальной позицией (pos_0)
    и желаемой позицией (pos_d) за время (T) в момент времени (t).
"""
def fifth_interpol(pos_0, pos_d, T, t):
    delta = pos_d - pos_0
    # a0 = 0
    # a1 = 0
    # a2 = 0
    a3 = 10 / T ** 3
    a4 = -15 / T ** 4
    a5 = 6 / T ** 5
    s = a5 * (t ** 5) + a4 * (t ** 4) + a3 * (t ** 3)

    return ((pos_0 + s * delta), delta * (3 * a3 * t ** 2 + 4 * a4 * t ** 3 + 5 * a5 * t ** 4),
            delta * (6 * a3 * t + 12 * a4 * t ** 2 + 20 * a5 * t ** 3)) if (t <= T) else (pos_d, 0, 0)

logTime = [t]
logPos = [q0_fact]
logPos_d = [q0_fact]
logVel = [0]
logVel_d = [0]
logAcc = [0]
logAcc_d = [0]
logCtrl = []

u = 0 # Инициализация управляющего воздействия

if (OWN_REG):
    p.setJointMotorControl2(bodyIndex=bodyId,
                            jointIndex=jointIdx,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=0,
                            force=0)

prev_vel = 0 # Предыдущая скорость

while t <= maxTime:
    pos = p.getJointState(bodyId, jointIdx)[0] # Получение текущей позиции маятника
    vel = p.getJointState(bodyId, jointIdx)[1] # Получение текущей скорости маятника

    # Вычисление желаемой позиции, скорости и ускорения с использованием функции fifth_interpol()
    (posd, veld, accd) = fifth_interpol(q0_fact, qd, trajTime, t)

    # Вычисление управляющего сигнала с использованием функции feedback_lin()
    u = feedback_lin(pos, vel, posd, veld, accd)

    if (OWN_REG):
        p.setJointMotorControl2(bodyIndex=bodyId,
                                jointIndex=jointIdx,
                                controlMode=p.TORQUE_CONTROL,
                                force=u)
    else:
        p.setJointMotorControl2(bodyIndex=bodyId,
                                jointIndex=jointIdx,
                                targetPosition=posd,
                                controlMode=p.POSITION_CONTROL)

    p.stepSimulation()
    t += dt

    logPos.append(pos)
    logVel.append(vel)
    logAcc.append((vel - prev_vel) / dt)
    logPos_d.append(posd)
    logVel_d.append(veld)
    logAcc_d.append(accd)
    prev_vel = vel
    logCtrl.append(u)
    logTime.append(t)

    if (GUI):
        time.sleep(dt)
p.disconnect()

import matplotlib.pyplot as plt

plt.subplot(4, 1, 1)
plt.plot(logTime, logPos, label='pos')
plt.plot(logTime, logPos_d, label='pos_d')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(logTime, logVel, label='vel')
plt.plot(logTime, logVel_d, label='vel_d')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(logTime, logAcc, label='acc')
plt.plot(logTime, logAcc_d, label='acc_d')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(logTime[0:-1], logCtrl, label='control')
plt.grid(True)
plt.legend()

plt.show()