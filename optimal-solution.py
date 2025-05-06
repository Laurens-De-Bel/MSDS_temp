import casadi as ca
from dynamics import lag_eq
import numpy as np
from functions import *
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint, Bounds
import time

begintime = time.time()
iterations = 0
# x_dot = openloop(x, u)
def openloop(x, u):
    global M, C
    q, q_d = x[0:5], x[5:10]
    #q_dd = ca.inv(M(q)) @ (B @ u - C(q, q_d))
    
    damping_coefficient = 0  # Experiment with this value
    q_dd = ca.inv(M(q)) @ (B @ u - C(q, q_d) - damping_coefficient * q_d)
    x_d = ca.vertcat(q_d, q_dd)
    return x_d

# Make sure to use DM for numeric evaluation in f:
def f(x, u):
    return openloop(x, u)


T, V, M, C, B = lag_eq([0,0])

# dimensions of the problem
Nt = 50 # Our horizon of the optimal control problem
Nx = 10  # number of state variables (q1...q5, q1d...q2d)
Nu = 4  # number of control variables (input torques)

dt = 0.01
t = np.arange(0, 3+dt, dt)

# choose steplength
L_step = 0.3

# Generate initial geuss
q0 = np.array([210, 140, 0, 0, 0, -50, 50, 10, 10, 10])*np.pi/180
qT = np.array(heelstrike_transformation(q0, M)).flatten()
U = np.ones((Nt, Nu))*0.1
X = np.linspace(q0, qT, Nt + 1)
# animate_solution(X,5*dt)

# We need to condense these vectors into one single vector P
p = np.hstack([X.reshape(-1),U.reshape(-1)])
# From this p, we get the state and control variables back like
ix = lambda k: slice(Nx*k,Nx*(k+1))
iu = lambda k: slice((Nt+1)*Nx+Nu*k,(Nt+1)*Nx+Nu*(k+1))

# Define the cost function
# l = lambda x,u: 1/2*np.linalg.norm(u)**2
l = lambda x,u: 1/2*(u[0]**2+u[1]**2+u[2]**2+u[3]**2)
L = lambda xk,xkk,uk: dt/2 * (l(xk,uk)+l(xkk, uk))
def cost(p):
    total_cost = 0
    
    for k in range(Nt):
        xk  = p[ix(k)]
        xkk = p[ix(k+1)]
        uk  = p[iu(k)]
        total_cost += L(xk,xkk,uk)
    return total_cost

# Continuity constraint
g = lambda xkk, xk, uk: xkk-xk-dt/2*(f(xk,uk)+f(xkk,uk))
def continuity_constraints(p):
    # we only put a continuity constraint on the states
    flat_vector = np.zeros(Nt*Nx) # What size should the result be?
    
    for k in range(Nt):
        xk  = p[ix(k)]
        xkk = p[ix(k+1)]
        uk  = p[iu(k)]
        flat_vector[Nx*k:Nx*(k+1)] = np.array(g(xk,xkk,uk)).flatten()
    return flat_vector

def casadi_continuity_constraints():
    # CasADi symbolic decision variables
    p = ca.MX.sym('p', (Nt+1)*Nx + Nt*Nu)

    constraints = []
    for k in range(Nt):
        xk  = p[ix(k)]
        xkk = p[ix(k+1)]
        uk  = p[iu(k)]

        # reshape to column vectors for CasADi
        xk  = ca.reshape(xk, (Nx, 1))
        xkk = ca.reshape(xkk, (Nx, 1))
        uk  = ca.reshape(uk, (Nu, 1))

        gk = g(xkk, xk, uk)  # CasADi will keep this symbolic
        constraints.append(gk)

    constraints_vec = ca.vertcat(*constraints)
    return p, constraints_vec

p_sym, g_sym = casadi_continuity_constraints()
jac_g = ca.jacobian(g_sym, p_sym)
dcontinuity_constraints_dp_fun = ca.Function('jac_g', [p_sym], [jac_g], ['p'], ['jac'])     # convert back to Python function
def dcontinuity_constraints_dp(p):
    global iterations
    J = dcontinuity_constraints_dp_fun(p)

    # print("Iteration:", iterations) 
    iterations +=1
    return J.full()

nonlinconstraint = NonlinearConstraint(continuity_constraints, np.zeros(Nt*Nx), np.zeros(Nt*Nx), jac=dcontinuity_constraints_dp)


# Initial state constraints
def initial_constr_eq(p):
    ret = []
    ret.append(float(get_coords(p[:10])[9][0]) + L_step)
    ret.append(float(get_coords(p[:10])[8][1]))
    return np.array(ret)

q = ca.MX.sym("q", 5)
q_dot = ca.MX.sym("q_dot", 5)
velocities = [ca.jacobian(pos, q) @ q_dot for pos in get_coords(q)]
get_coords_velocity = ca.Function("get_coords_velocity", [q, q_dot], [ca.vertcat(*velocities)])
def initial_constr_ineq(p):
    x0 = p[0:10]                # initial state: q0 and q0_dot
    q0 = x0[:5]
    qd0 = x0[5:]
    vels = get_coords_velocity(q0, qd0)
    v_P5 = vels[9*2 : 9*2+2]    # extract x and y velocity of P5
    vy_P5 = float(v_P5[1])      # y velocity
    return np.atleast_1d(vy_P5)


initial_constr1 = NonlinearConstraint(initial_constr_eq, np.zeros(2),np.zeros(2))
initial_constr2 = NonlinearConstraint(initial_constr_ineq, 0.0000001, 100000)

# Final state constraints
def final_constr_eq(p):
    ret = []
    ret.append(float(get_coords(p[-14:-4])[9][0]) - L_step)
    ret.append(float(get_coords(p[-14:-4])[8][1]))
    return np.array(ret)

def final_constr_ineq(p):
    x0 = p[-14:-4]                # initial state: q0 and q0_dot
    q0 = x0[:5]
    qd0 = x0[5:]
    vels = get_coords_velocity(q0, qd0)
    v_P5 = vels[9*2 : 9*2+2]    # extract x and y velocity of P5
    vy_P5 = float(v_P5[1])      # y velocity
    return np.atleast_1d(vy_P5)

final_constr1 = NonlinearConstraint(final_constr_eq, np.zeros(2),np.zeros(2))
final_constr2 = NonlinearConstraint(final_constr_ineq, -1000000, 0.0000001)

# ground constraints
def P3_on_ground(p):
    ret = []
    P3_initial = get_coords(p[ix(0)])[7]
    for i in range(Nt):
        P3 = get_coords(p[ix(i)])[7]  # returns a 2D point
        x = float(P3[0]) - float(P3_initial[0]) # convert CasADi or numpy scalar to Python float
        y = float(P3[1])
        ret.extend([x, y])
    return np.array(ret)

def P4_above_ground(p):
    ret = []
    for i in range(Nt):
        P4 = get_coords(p[ix(i)])[8]  # returns a 2D point
        y = float(P4[1])
        ret.append(y)
    return np.array(ret)

P3_fixed_constr = NonlinearConstraint(P3_on_ground, np.zeros(2*Nt),np.zeros(2*Nt))
P4_ground_constr = NonlinearConstraint(P4_above_ground, 0.05*np.ones(Nt),1000*np.ones(Nt))

# Left-right symmetry
def left_right_symm(p):
    global M
    ret = p[ix(0)] - heelstrike_transformation(p[ix(Nt)], M) 
    return np.atleast_1d(np.array(ret).flatten())

symm_constr = NonlinearConstraint(left_right_symm, np.zeros(10),np.zeros(10))


# bounds
qmax = np.array([7, 7, 7, 7, 7, 20, 20,20,20,20])#*np.pi/180
qmin = -qmax
umax = np.array([20,20,20,20]) 
umin = -umax
q_upperbound = np.tile(qmax, Nt+1)
u_upperbound = np.tile(umax, Nt)
p_upperbound = np.hstack([q_upperbound,u_upperbound])
p_lowerbound = -p_upperbound
bounds = Bounds(p_lowerbound,p_upperbound)



# Compute solution
res = minimize(cost, p, constraints=(nonlinconstraint, initial_constr2, final_constr1, P3_fixed_constr, P4_ground_constr, symm_constr),options=dict(maxiter=20,gtol=1e-3,xtol=1e-3,barrier_tol=1e-3,verbose=3), method='trust-constr')
# res = minimize(cost,  p, constraints=(nonlinconstraint, symm_constr, P4_ground_constr),options=dict(maxiter=200,ftol=1e-10, disp = True), method='SLSQP')
# assert res.success        

# Xdms = res.x[:(Nt+1)*Nx].reshape(-1,Nx)
Qdms = res.x[: (Nt + 1) * Nx].reshape((Nt + 1, Nx))
Udms = res.x[(Nt+1)*Nx:].reshape(-1,Nu)

# print(Qdms)
# print(Udms)
print('The calculation took', time.time()-begintime, 'seconds')
print('Begin generating animation now...')
animate_solution(Qdms,5*dt)