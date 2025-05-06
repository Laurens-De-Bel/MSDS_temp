import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Given variables
g = 9.81                                                #[m/s^2]
m1, m2, m3, m4, m5 = 6.8, 6.8, 3.2, 3.2, 20             #[kg]
l1, l2, l3, l4, l5 = 0.4, 0.4, 0.4, 0.4, 0.625          #[m]
r1, r2, r3, r4, r5 = 0.163, 0.163, 0.128, 0.128, 0.2    #[m]
I1, I2, I3, I4, I5 = 1.08, 1.08, 0.93, 0.93, 2.22       #[kg m^2]

def get_coords(x):
    q = x[:5]
    # Define angles
    alpha = q[0] + q[4] - ca.pi/2
    beta = q[1] + q[4] - ca.pi
    gamma = q[0] + q[2] + q[4] -ca.pi/2
    delta = q[1] - q[3] + q[4] -ca.pi


    # Define the position vectors
    P3 = ca.vertcat(x[5],0)

    G3 = ca.vertcat(P3[0] + (l3-r3) * ca.cos(gamma), P3[1] + (l3-r3) * ca.sin(gamma))
    P1 = ca.vertcat(P3[0] + l3*ca.cos(gamma), P3[1] + l3*ca.sin(gamma))

    G1 = ca.vertcat(P1[0] + (l1-r1)*ca.cos(alpha), P1[1] + (l1-r1)*ca.sin(alpha))
    P5 = ca.vertcat(P1[0] + l1*ca.cos(alpha), P1[1] + l1*ca.sin(alpha))

    G5 = ca.vertcat(P5[0] + r5*ca.cos(q[4] + ca.pi/2), P5[1] + r5*ca.sin(q[4] + ca.pi/2))
    P6 = ca.vertcat(P5[0] + l5*ca.cos(q[4] + ca.pi/2), P5[1] + l5*ca.sin(q[4] + ca.pi/2))

    G2 = ca.vertcat(P5[0] + r2*ca.sin(beta), P5[1] - r2*ca.cos(beta))
    P2 = ca.vertcat(P5[0] + l2*ca.sin(beta), P5[1] - l2*ca.cos(beta))

    G4 = ca.vertcat(P2[0] + r4*ca.sin(delta), P2[1] - r4*ca.cos(delta))
    P4 = ca.vertcat(P2[0] + l4*ca.sin(delta), P2[1] - l4*ca.cos(delta))

    return G1, G2, G3, G4, G5, P1, P2, P3, P4, P5, P6


def heelstrike(x):
    P4 = get_coords(x)[8]
    return P4[1] < 0.0

def heelstrike_transformation(x_min, M):
    q_min = x_min[:5]
    qd_min = x_min[5:]

    # q_plus = ca.DM([[0,1,0,0,0],[1,0,0,0,0],[0,0,0,1,0],[0,0,1,0,0],[0,0,0,0,1]])@q_min
    q_plus = ca.vertcat(q_min[1], q_min[0], q_min[3], q_min[2], q_min[4])

    M_min = M(q_min)
    M_plus = M(q_plus)
    qd_plus = ca.solve(M_plus, M_min @ qd_min)

    return ca.vertcat(q_plus[0], q_plus[1], q_plus[2], q_plus[3], q_plus[4], qd_plus[0], qd_plus[1], qd_plus[2], qd_plus[3], qd_plus[4])


# Animation function
def animate_solution(x_solution, dt):

    fig, ax = plt.subplots()
    ax.set_title('Mechanism Animation')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    # Adjust these limits as needed based on your mechanism's reach
    ax.set_xlim(-1, 7)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')

    # Use a single color for all parts; change 'red' to any other color if desired.
    color = 'red'

    # Prepare line objects for the two "arms" and centers of mass, all with the same color
    # Arm 1: P3 -> P1 -> P5 -> P6
    line1, = ax.plot([], [], 'o-', color=color, lw=2, label='Arm 1')
    # Arm 2: P5 -> P2 -> P4
    line2, = ax.plot([], [], 'o-', color=color, lw=2, label='Arm 2')
    # Centers of mass (G1 to G5) as markers with the same color
    com_points, = ax.plot([], [], 'o', color=color, label='Centers of Mass')

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        com_points.set_data([], [])
        return line1, line2, com_points

    def update(frame):
        # Extract the state at the current timestep
        x = x_solution[frame]
        # Compute Cartesian coordinates using get_coords
        G1, G2, G3, G4, G5, P1, P2, P3, P4, P5, P6 = get_coords(x)
        
        # Helper: convert CasADi DM to a flat NumPy array
        to_np = lambda vec: np.array(vec.full()).flatten()

        # Convert key points
        P3_np = to_np(P3)
        P1_np = to_np(P1)
        P5_np = to_np(P5)
        P6_np = to_np(P6)
        P2_np = to_np(P2)
        P4_np = to_np(P4)
        
        G1_np = to_np(G1)
        G2_np = to_np(G2)
        G3_np = to_np(G3)
        G4_np = to_np(G4)
        G5_np = to_np(G5)
        
        # Update Arm 1: connecting P3 -> P1 -> P5 -> P6
        xdata_arm1 = [P3_np[0], P1_np[0], P5_np[0], P6_np[0]]
        ydata_arm1 = [P3_np[1], P1_np[1], P5_np[1], P6_np[1]]
        line1.set_data(xdata_arm1, ydata_arm1)
        
        # Update Arm 2: connecting P5 -> P2 -> P4
        xdata_arm2 = [P5_np[0], P2_np[0], P4_np[0]]
        ydata_arm2 = [P5_np[1], P2_np[1], P4_np[1]]
        line2.set_data(xdata_arm2, ydata_arm2)
        
        # Update centers of mass: G1, G2, G3, G4, G5
        xdata_com = [G1_np[0], G2_np[0], G3_np[0], G4_np[0], G5_np[0]]
        ydata_com = [G1_np[1], G2_np[1], G3_np[1], G4_np[1], G5_np[1]]
        com_points.set_data(xdata_com, ydata_com)
        
        return line1, line2, com_points

    ani = animation.FuncAnimation(
        fig, update, frames=len(x_solution), init_func=init,
        blit=True, interval=dt * 1000  # interval in milliseconds
    )
    
    # Save the animation as a GIF using the PillowWriter
    fps = int(1 / dt)
    ani.save('mechanism_animation_heelstrike.gif', writer='pillow', fps=fps)
    
    plt.show()

def plot_energies(T_list, V_list):
    T_sol = np.array([ti.full() for ti in T_list])
    V_sol = np.array([vi.full() for vi in V_list])
    E_sol = T_sol+V_sol
    t = []
    v = []
    tot = []
    for i in T_sol:
        for e in i:
            t.append(float(e[0]))
    for i in V_sol:
        for e in i:
            v.append(float(e[0]))
    for i in E_sol:
        for e in i:
            tot.append(float(e[0]))

    plt.plot(t)
    plt.plot(v)
    plt.plot(tot)
    plt.title('blauw = E_kin, oranje = E_pot, groen = E_tot')
    plt.show()