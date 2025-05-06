import casadi as ca
import numpy as np

def lag_eq(_P3):
    # Given variables
    g = 9.81                                                #[m/s^2]
    m1, m2, m3, m4, m5 = 6.8, 6.8, 3.2, 3.2, 20             #[kg]
    l1, l2, l3, l4, l5 = 0.4, 0.4, 0.4, 0.4, 0.625          #[m]
    r1, r2, r3, r4, r5 = 0.163, 0.163, 0.128, 0.128, 0.2    #[m]
    I1, I2, I3, I4, I5 = 1.08, 1.08, 0.93, 0.93, 2.22       #[kg m^2]

    # Define the generalized coordinates
    q1, q2, q3, q4, q5 = ca.MX.sym('q1'), ca.MX.sym('q2'), ca.MX.sym('q3'), ca.MX.sym('q4'), ca.MX.sym('q5')
    q1_d, q2_d, q3_d, q4_d, q5_d = ca.MX.sym('q1_d'), ca.MX.sym('q2_d'), ca.MX.sym('q3_d'), ca.MX.sym('q4_d'), ca.MX.sym('q5_d')
    q = ca.vertcat(q1, q2, q3, q4, q5)
    q_d = ca.vertcat(q1_d, q2_d, q3_d, q4_d, q5_d)

    # Define angles
    alpha = q[0] + q[4] - ca.pi/2
    beta = q[1] + q[4] - ca.pi
    gamma = q[0] + q[2] + q[4] -ca.pi/2
    delta = q[1] - q[3] + q[4] -ca.pi

    # Define the position vectors
    P3 = _P3

    G3 = ca.vertcat(P3[0]+(l3-r3) * ca.cos(gamma), P3[1] + (l3-r3) * ca.sin(gamma))
    P1 = ca.vertcat(P3[0]+l3*ca.cos(gamma),P3[1] + l3*ca.sin(gamma))

    G1 = ca.vertcat(P1[0] + (l1-r1)*ca.cos(alpha), P1[1] + (l1-r1)*ca.sin(alpha))
    P5 = ca.vertcat(P1[0] + l1*ca.cos(alpha), P1[1] + l1*ca.sin(alpha))

    G5 = ca.vertcat(P5[0] + r5*ca.cos(q[4] + ca.pi/2), P5[1] + r5*ca.sin(q[4] + ca.pi/2))
    P6 = ca.vertcat(P5[0] + l5*ca.cos(q[4] + ca.pi/2), P5[1] + l5*ca.sin(q[4] + ca.pi/2))

    G2 = ca.vertcat(P5[0] + r2*ca.sin(beta), P5[1] - r2*ca.cos(beta))
    P2 = ca.vertcat(P5[0] + l2*ca.sin(beta), P5[1] - l2*ca.cos(beta))

    G4 = ca.vertcat(P2[0] + r4*ca.sin(delta), P2[1] - r4*ca.cos(delta))
    P4 = ca.vertcat(P2[0] + l4*ca.sin(delta), P2[1] - l4*ca.cos(delta))

    # Velocity vectors
    # Derive Gi with respect to time, so due to chain rule, this becomes the following:
    G1_v = ca.jacobian(G1, q)@q_d
    G2_v = ca.jacobian(G2, q)@q_d
    G3_v = ca.jacobian(G3, q)@q_d
    G4_v = ca.jacobian(G4, q)@q_d
    G5_v = ca.jacobian(G5, q)@q_d

    # Kinetic energy
    T_lin = 1/2*(m1*(G1_v[0]**2+G1_v[1]**2)+m2*(G2_v[0]**2+G2_v[1]**2)+m3*(G3_v[0]**2+G3_v[1]**2)+m4*(G4_v[0]**2+G4_v[1]**2)+m5*(G5_v[0]**2+G5_v[1]**2))
    T_rot = 1/2*(I1*q1_d**2 + I2*q2_d**2 + I3*q3_d**2 + I4*q4_d**2 + I5*q5_d**2)
    T = T_lin+T_rot

    # Potential energy
    V = g*(m1*G1[1] + m2*G2[1] + m3*G3[1] + m4*G4[1] + m5*G5[1])

    # Euler-Lagrange equations
    L = T-V

    L_q = ca.jacobian(L,q)  # =g
    L_qd = ca.jacobian(L,q_d)

    M = ca.jacobian(L_qd,q_d)
    S = ca.jacobian(L_qd, q)
    C = S@q_d-L_q.T

    T = ca.Function('T', [q, q_d], [T])
    V = ca.Function('V', [q], [V])
    M = ca.Function('M', [q], [M])
    C = ca.Function('C', [q, q_d], [C])
    B = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [0,0,0,0]])


    return T,V,M,C,B