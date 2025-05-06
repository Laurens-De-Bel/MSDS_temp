import casadi as ca
from dynamics import lag_eq
import numpy as np
from functions import *

# x_dot = openloop(x, u)
def openloop(x, u):
    global M, C
    q, q_d = x[:5], x[5:]
    #q_dd = ca.inv(M(q)) @ (B @ u - C(q, q_d))
    
    damping_coefficient = 0  # Experiment with this value
    if Frozen_body:
        q_d[:4] = 0
        m = M(q)
        c = C(q,q_d)
        q_dd = [ 0, 0, 0, 0, float(-c[-1] / m[-1, -1] - damping_coefficient*q_d[-1]) ] 
    else:
        q_dd = ca.inv(M(q)) @ (B @ u - C(q, q_d) - damping_coefficient * q_d)
    x_d = ca.vertcat(q_d, q_dd)
    return x_d

# Make sure to use DM for numeric evaluation in f:
def f(x, t):
    u = ca.DM.zeros(4, 1)  # Use DM instead of MX for numeric zeros
    return openloop(x, u)

# Choose in which way you want to validate the model
Frozen_body = True

# Define the generalized coordinates (same as before)
q1, q2, q3, q4, q5 = ca.MX.sym('q1'), ca.MX.sym('q2'), ca.MX.sym('q3'), ca.MX.sym('q4'), ca.MX.sym('q5')
q1_d, q2_d, q3_d, q4_d, q5_d = ca.MX.sym('q1_d'), ca.MX.sym('q2_d'), ca.MX.sym('q3_d'), ca.MX.sym('q4_d'), ca.MX.sym('q5_d')
q = ca.vertcat(q1, q2, q3, q4, q5)
q_d = ca.vertcat(q1_d, q2_d, q3_d, q4_d, q5_d)
x = ca.vertcat(q, q_d)

# Define the input torques
u1, u2, u3, u4 = ca.MX.sym('u1'), ca.MX.sym('u2'), ca.MX.sym('u3'), ca.MX.sym('u4')
u = ca.vertcat(u1, u2, u3, u4)

# M and C matrix from Euler-Lagrange
T, V, M, C, B = lag_eq([0,0])

dt = 0.01
t = np.arange(0, 3+dt, dt)
if Frozen_body:
    q0 = np.array([220, 270, -60, 0, -25, 0, 0, 0, 0, -100])*np.pi/180
else:
    q0 = np.array([220, 270, -60, 0, -25, 0, 0, 0, 0, 0])*np.pi/180
x = ca.DM(q0)  # Ensure x is a DM for numeric computation
X = []
number_of_heelstrikes = 0
P3 = [0,0]
T_list = []
V_list = []
time_of_last_heelstrike = 0

# Runge-Kutta 4 is used for integration
for _t in t:
    k1 = f(x, _t)
    k2 = f(x + dt/2 * k1, _t + dt/2)
    k3 = f(x + dt/2 * k2, _t + dt/2)
    k4 = f(x + dt * k3, _t + dt)
    x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    if Frozen_body:
        x[5:9] = 0
    
    T_list.append(T(x[:5], x[5:]))
    V_list.append(V(x[:5]))

    # T_list.append(T(ca.DM([x[0], x[1], x[2], -x[3], x[4]]), ca.DM([x[5], x[6], x[7], -x[8], x[9]])))
    # V_list.append(V(ca.DM([x[0], x[1], x[2], -x[3], x[4]])))

    if Frozen_body and heelstrike([x[0], x[1], x[2], x[3], x[4], P3[0], P3[1], 0, 0, 0]) and number_of_heelstrikes < 8 and _t-time_of_last_heelstrike > 10*dt:
        # This detects whether a heelstrike is performed
        if number_of_heelstrikes%2 == 0:
            P3 = get_coords(ca.DM([x[0], x[1], x[2], x[3], x[4], P3[0], P3[1], 0, 0, 0]))[8]
            P3[1] = 0
            print('heelstrike!')
            T, V, M, C, B = lag_eq(P3)
        else:
            P3 = get_coords(ca.DM([x[0], x[1], x[2], -x[3], x[4], P3[0], P3[1], 0, 0, 0]))[8]
            P3[1] = 0
            print('heelstrike!')
            T, V, M, C, B = lag_eq(P3)
        number_of_heelstrikes +=1
        time_of_last_heelstrike = _t
        x = heelstrike_transformation(x, M)
        
    # print(P3)
    # append x to the output list X
    if number_of_heelstrikes%2 == 0:
        X.append(ca.DM([x[0], x[1], x[2], x[3], x[4], P3[0], P3[1], 0, 0, 0]))  # Each x is a DM, so conversion later will work
    else:
        # X.append(ca.DM([x[1], x[0], x[3], x[2], x[4], P0, x[6], x[7], x[8], x[9]]))
        X.append(ca.DM([x[0], x[1], x[2], -x[3], x[4], P3[0], P3[1], 0, 0, 0]))




# Convert the list of DMs to a NumPy array:
x_solution = np.array([xi.full() for xi in X])
# print(x_solution)

plot_energies(T_list, V_list)

animate_solution(x_solution,dt)