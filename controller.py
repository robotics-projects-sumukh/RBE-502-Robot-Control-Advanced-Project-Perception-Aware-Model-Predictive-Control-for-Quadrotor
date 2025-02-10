import os
import casadi as cs
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from quadrotor import Quadrotor3D
from utils import skew_symmetric, v_dot_q, quaternion_inverse, q_dot_q


class Controller:
    def __init__(self, quad:Quadrotor3D, t_horizon=1, n_nodes=20,
                 q_cost=None, r_cost=None, q_mask=None, rdrv_d_mat=None,
                 model_name="quad_3d_acados_mpc", solver_options=None):
        """
        :param quad: quadrotor object
        :type quad: Quadrotor3D
        :param t_horizon: time horizon for MPC optimization
        :param n_nodes: number of optimization nodes until time horizon
        :param q_cost: diagonal of Q matrix for LQR cost of MPC cost function. Must be a numpy array of length 15.
        :param r_cost: diagonal of R matrix for LQR cost of MPC cost function. Must be a numpy array of length 4.
        :param q_mask: Optional boolean mask that determines which variables from the state compute towards the cost
        function. In case no argument is passed, all variables are weighted.
        :param solver_options: Optional set of extra options dictionary for solvers.
        :param rdrv_d_mat: 3x3 matrix that corrects the drag with a linear model according to Faessler et al. 2018. None
        if not used
        """

        # Weighted squared error loss function q = (p_xyz, a_xyz, v_xyz, s_xyz, s_dot_xyz), r = (u1, u2, u3, u4)
        if q_cost is None:
            q_cost = np.array([10, 10, 10, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 100, 100, 100, 1, 1, 1])
        if r_cost is None:
            r_cost = np.array([0.1, 0.1, 0.1, 0.1])

        self.T = t_horizon  # Time horizon
        self.N = n_nodes  # number of control nodes within horizon

        self.quad = quad

        self.max_u = quad.max_input_value
        self.min_u = quad.min_input_value

        self.max_thrust = quad.max_thrust

        self.min_linear_velocity = np.array([-3.0, -3.0, -3.0])  # Min linear velocity in x, y, z
        self.max_linear_velocity = np.array([3.0, 3.0, 3.0])     # Max linear velocity in x, y, z

        self.min_angular_velocity = np.array([-2.0, -2.0, -2.0])  # Min angular velocity in roll, pitch, yaw
        self.max_angular_velocity = np.array([2.0, 2.0, 2.0])     # Max angular velocity in roll, pitch, yaw

        self.max_pos = np.array([10000, 10000, 10000])  # Max position in x, y, z
        self.min_pos = np.array([-10000, -10000, -10000])  # Min position in x, y, z

        # Declare model variables
        self.p = cs.MX.sym('p', 3)  # position
        self.q = cs.MX.sym('q', 4)  # angle quaternion (wxyz)
        self.v = cs.MX.sym('v', 3)  # velocity
        self.s = cs.MX.sym('s', 3)  # projection of point of interest
        self.s_dot = cs.MX.sym('s_dot', 3)  # projection_derivative of point of interest

        # Full state vector (16-dimensional)
        self.x = cs.vertcat(self.p, self.q, self.v, self.s, self.s_dot)
        self.state_dim = 16

        # Control input vector
        self.c = cs.MX.sym('c') # thrust
        self.wx = cs.MX.sym('wx') # angular velocity x
        self.wy = cs.MX.sym('wy') # angular velocity y
        self.wz = cs.MX.sym('wz') # angular velocity z
        self.u = cs.vertcat(self.c, self.wx, self.wy, self.wz)

        # Parameters for calculation of Projection
        self.Wpf = cs.MX(quad.Wpf.tolist())
        self.pBC = cs.MX(quad.pBC.tolist())
        self.qBC = cs.MX(quad.qBC.tolist())
        self.Cpf = cs.MX(quad.Cpf.tolist())
        self.Cp_dot_f = cs.MX(quad.Cp_dot_f.tolist())

        self.fx = cs.MX(quad.fx)
        self.fy = cs.MX(quad.fy)

        # Nominal model equations symbolic function (no GP)
        self.quad_xdot_nominal = self.quad_dynamics(rdrv_d_mat)

        # Initialize objective function, 0 target state and integration equations
        self.L = None
        self.target = None

        # Build full model. Will have 16 variables. self.dyn_x contains the symbolic variable that should be used to evaluate the dynamics function. It corresponds to self.x if there are no GP's, or self.x_with_gp otherwise
        acados_models, nominal_with_gp = self.acados_setup_model(
            self.quad_xdot_nominal(x=self.x, u=self.u)['x_dot'], model_name)

        # Convert dynamics variables to functions of the state and input vectors
        self.quad_xdot = {}
        for dyn_model_idx in nominal_with_gp.keys():
            dyn = nominal_with_gp[dyn_model_idx]
            self.quad_xdot[dyn_model_idx] = cs.Function('x_dot', [self.x, self.u], [dyn], ['x', 'u'], ['x_dot'])

        # ### Setup and compile Acados OCP solvers ### #
        self.acados_ocp_solver = {}

        # Add one more weight to the rotation (use quaternion norm weighting in acados)
        q_diagonal = np.concatenate((q_cost[:3], np.mean(q_cost[3:6])[np.newaxis], q_cost[3:]))
        if q_mask is not None:
            q_mask = np.concatenate((q_mask[:3], np.zeros(1), q_mask[3:]))
            q_diagonal *= q_mask

        for key, key_model in zip(acados_models.keys(), acados_models.values()):
            nx = key_model.x.size()[0]
            nu = key_model.u.size()[0]
            ny = nx + nu
            n_param = key_model.p.size()[0] if isinstance(key_model.p, cs.MX) else 0

            # Create OCP object to formulate the optimization
            ocp = AcadosOcp()
            ocp.model = key_model

            # Print the explicit dynamics function
            # print(ocp.model.f_expl_expr)
            # input()

            ocp.dims.N = self.N
            ocp.solver_options.tf = t_horizon

            # Initialize parameters
            ocp.dims.np = n_param
            ocp.parameter_values = np.zeros(n_param)

            ocp.cost.cost_type = 'LINEAR_LS'
            ocp.cost.cost_type_e = 'LINEAR_LS'

            ocp.cost.W = np.diag(np.concatenate((q_diagonal, r_cost)))
            ocp.cost.W_e = np.diag(q_diagonal)
            terminal_cost = 0 if solver_options is None or not solver_options["terminal_cost"] else 1
            ocp.cost.W_e *= terminal_cost

            ocp.cost.Vx = np.zeros((ny, nx))
            ocp.cost.Vx[:nx, :nx] = np.eye(nx)
            ocp.cost.Vu = np.zeros((ny, nu))
            ocp.cost.Vu[-4:, -4:] = np.eye(nu)

            ocp.cost.Vx_e = np.eye(nx)

            # Initial reference trajectory (will be overwritten)
            x_ref = np.zeros(nx)
            ocp.cost.yref = np.concatenate((x_ref, np.array([0.0, 0.0, 0.0, 0.0])))
            ocp.cost.yref_e = x_ref

            # Initial state (will be overwritten)
            ocp.constraints.x0 = x_ref

            # Set constraints on control input (index 0 in control vector)
            ocp.constraints.lbu = np.array([self.min_u])
            ocp.constraints.ubu = np.array([self.max_u])
            ocp.constraints.idxbu = np.array([0])

            # Set constraints on angular velocity  (indices 1:4 in control vector)
            ocp.constraints.lbu = np.concatenate((ocp.constraints.lbu, self.min_angular_velocity))
            ocp.constraints.ubu = np.concatenate((ocp.constraints.ubu, self.max_angular_velocity))
            ocp.constraints.idxbu = np.concatenate((ocp.constraints.idxbu, np.array([1, 2, 3])))

            # Set constraints on position (indices 0:3 in state vector)
            ocp.constraints.lbx = np.concatenate((ocp.constraints.lbx, self.min_pos))
            ocp.constraints.ubx = np.concatenate((ocp.constraints.ubx, self.max_pos))
            ocp.constraints.idxbx = np.concatenate((ocp.constraints.idxbx, np.array([0, 1, 2])))

            # Set constraints on quaternion (indices 3:7 in state vector)
            ocp.constraints.lbx = np.concatenate((ocp.constraints.lbx, np.array([-2, -2, -2, -2])))
            ocp.constraints.ubx = np.concatenate((ocp.constraints.ubx, np.array([2, 2, 2, 2])))
            ocp.constraints.idxbx = np.concatenate((ocp.constraints.idxbx, np.array([3, 4, 5, 6])))           

            # Set constraints on linear velocity (indices 7:10 in state vector) 
            ocp.constraints.lbx = np.concatenate((ocp.constraints.lbx, self.min_linear_velocity))
            ocp.constraints.ubx = np.concatenate((ocp.constraints.ubx, self.max_linear_velocity))
            ocp.constraints.idxbx = np.concatenate((ocp.constraints.idxbx, np.array([7, 8, 9])))

            # Solver options
            ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
            ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
            ocp.solver_options.integrator_type = 'ERK'
            ocp.solver_options.print_level = 0
            # ocp.solver_options.nlp_solver_type = 'SQP_RTI' if solver_options is None else solver_options["solver_type"]
            ocp.solver_options.nlp_solver_type = 'SQP_RTI'

            # Compile acados OCP solver if necessary
            json_file = os.path.join('./', key_model.name + '_acados_ocp.json')
            self.acados_ocp_solver[key] = AcadosOcpSolver(ocp, json_file=json_file)

    def acados_setup_model(self, nominal, model_name):
        """
        Builds an Acados symbolic models using CasADi expressions.
        :param model_name: name for the acados model. Must be different from previously used names or there may be
        problems loading the right model.
        :param nominal: CasADi symbolic nominal model of the quadrotor: f(self.x, self.u) = x_dot, dimensions 13x1.
        :return: Returns a total of three outputs, where m is the number of GP's in the GP ensemble, or 1 if no GP:
            - A dictionary of m AcadosModel of the GP-augmented quadrotor
            - A dictionary of m CasADi symbolic nominal dynamics equations with GP mean value augmentations (if with GP)
        :rtype: dict, dict, cs.MX
        """

        def fill_in_acados_model(x, u, p, dynamics, name):

            x_dot = cs.MX.sym('x_dot', dynamics.shape)
            f_impl = x_dot - dynamics

            # print(f_impl)
            # input()

            # Dynamics model
            model = AcadosModel()
            model.f_expl_expr = dynamics
            model.f_impl_expr = f_impl
            model.x = x
            model.xdot = x_dot
            model.u = u
            model.p = p
            model.name = name

            return model

        acados_models = {}
        dynamics_equations = {}

        # No available GP so return nominal dynamics
        dynamics_equations[0] = nominal

        x_ = self.x
        dynamics_ = nominal

        acados_models[0] = fill_in_acados_model(x=x_, u=self.u, p=[], dynamics=dynamics_, name=model_name)

        return acados_models, dynamics_equations

    def quad_dynamics(self, rdrv_d):
        """
        Symbolic dynamics of the 3D quadrotor model. The state consists on: [p_xyz, a_wxyz, v_xyz, r_xyz]^T, where p
        stands for position, a for angle (in quaternion form), v for velocity and r for body rate. The input of the
        system is: [u_1, u_2, u_3, u_4], i.e. the activation of the four thrusters.

        :param rdrv_d: a 3x3 diagonal matrix containing the D matrix coefficients for a linear drag model as proposed
        by Faessler et al.

        :return: CasADi function that computes the analytical differential state dynamics of the quadrotor model.
        Inputs: 'x' state of quadrotor (6x1) and 'u' control input (2x1). Output: differential state vector 'x_dot'
        (6x1)
        """

        x_dot = cs.vertcat(self.p_dynamics(), self.q_dynamics(), self.v_dynamics(rdrv_d), self.s_dynamics(), self.s_dot_dynamics())
        return cs.Function('x_dot', [self.x[:16], self.u], [x_dot], ['x', 'u'], ['x_dot'])

    def p_dynamics(self):
        return self.v

    def q_dynamics(self):
        return 1 / 2 * cs.mtimes(skew_symmetric(self.u[1:]), self.q)

    def v_dynamics(self, rdrv_d):
        """
        :param rdrv_d: a 3x3 diagonal matrix containing the D matrix coefficients for a linear drag model as proposed
        by Faessler et al. None, if no linear compensation is to be used.
        """

        f_thrust = self.u[0] * self.max_thrust
        g = cs.vertcat(0.0, 0.0, 9.81)
        a_thrust = cs.vertcat(0.0, 0.0, f_thrust)

        v_dynamics = v_dot_q(a_thrust, self.q) - g

        if rdrv_d is not None:
            # Velocity in body frame:
            v_b = v_dot_q(self.v, quaternion_inverse(self.q))
            rdrv_drag = v_dot_q(cs.mtimes(rdrv_d, v_b), self.q)
            v_dynamics += rdrv_drag

        return v_dynamics
    
    def s_dynamics(self):
        Wpf = cs.MX([5, 5, 5])
        pBC = cs.MX([0, 0, 0])
        qBC = cs.MX([0.7071, 0, 0.7071, 0])
        fx = cs.MX(0.016*640)
        fy = cs.MX(0.016*640)

        # Cpf = v_dot_q(self.Wpf - (v_dot_q(self.pBC, self.q) + self.p), quaternion_inverse(q_dot_q(self.q, self.qBC)))
        Cpf = v_dot_q(Wpf - (v_dot_q(pBC, self.q) + self.p), quaternion_inverse(q_dot_q(self.q, qBC)))
    
        s  = cs.vertcat(
            fx * Cpf[0] / Cpf[2],
            fy * Cpf[1] / Cpf[2],
            0)
        
        return s
    
    def s_dot_dynamics(self):

        Wpf = cs.MX([5, 5, 5])
        pBC = cs.MX([0, 0, 0])
        qBC = cs.MX([0.7071, 0, 0.7071, 0])
        fx = cs.MX(0.016*640)
        fy = cs.MX(0.016*640)

        Cpf = v_dot_q(Wpf - (v_dot_q(pBC, self.q) + self.p), quaternion_inverse(q_dot_q(self.q, qBC)))

        omega_c = v_dot_q(self.u[1:], quaternion_inverse(qBC))

        skew_omega_B = cs.vertcat(
            cs.horzcat(0, -self.u[3], self.u[2]),
            cs.horzcat(self.u[3], 0, -self.u[1]),
            cs.horzcat(-self.u[2], self.u[1], 0))
        
        CvWC = v_dot_q((0.5*cs.mtimes(skew_omega_B, v_dot_q(pBC, self.q)) + self.v), quaternion_inverse(q_dot_q(self.q, qBC)))

        skew_omega_C = cs.vertcat(
            cs.horzcat(0, -omega_c[2], omega_c[1]),
            cs.horzcat(omega_c[2], 0, -omega_c[0]),
            cs.horzcat(-omega_c[1], omega_c[0], 0))
        
        Cp_dot_f = - 0.5*cs.mtimes(skew_omega_C, Cpf) - CvWC

        s_dot = cs.vertcat(
            fx/Cpf[2]**2 * (Cpf[2]*Cp_dot_f[0] - Cpf[0]*Cp_dot_f[2]),
            -fy/Cpf[2]**2 * (Cpf[1]*Cp_dot_f[2] - Cpf[2]*Cp_dot_f[1]),
            0)

        return s_dot
    

    def run_optimization(self, initial_state=None, goal=None, use_model=0, return_x=False, mode='pose'):
        """
        Optimizes a trajectory to reach the pre-set target state, starting from the input initial state, that minimizes
        the quadratic cost function and respects the constraints of the system

        :param initial_state: 16-element list of the initial state. If None, 0 state will be used
        :param goal: 3 element [x,y,z] for moving to goal mode, 3*(N+1) for trajectory tracking mode
        :param use_model: integer, select which model to use from the available options.
        :param return_x: bool, whether to also return the optimized sequence of states alongside with the controls.
        :param mode: string, whether to use moving to pose mode or tracking mode
        :return: optimized control input sequence (flattened)
        """

        if initial_state is None:
            initial_state = [0, 0, 0] + [1, 0, 0, 0] + [0, 0, 0] + [0, 0, 0] + [0, 0, 0] 

        # Set initial state
        x_init = initial_state
        x_init = np.stack(x_init)

        # Set initial condition, equality constraint
        self.acados_ocp_solver[use_model].set(0, 'lbx', x_init)
        self.acados_ocp_solver[use_model].set(0, 'ubx', x_init)

        # Set final condition
        if mode == "pose":
            for j in range(self.N):
                y_ref = np.array([goal[0], goal[1], goal[2], 1,0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,0])
                self.acados_ocp_solver[use_model].set(j, 'yref', y_ref)
            y_refN = np.array([goal[0], goal[1], goal[2], 1,0,0,0, 0,0,0, 0,0,0, 0,0,0])
            self.acados_ocp_solver[use_model].set(self.N, 'yref', y_refN)
        else:
            for j in range(self.N):
                y_ref = np.array([goal[j,0], goal[j,1], goal[j,2], 1,0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,0])
                # y_ref = np.array([x_init[0], x_init[1], x_init[2], 1,0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,0])
                self.acados_ocp_solver[use_model].set(j, 'yref', y_ref)
            y_refN = np.array([goal[self.N,0], goal[self.N,1], goal[self.N,2], 1,0,0,0, 0,0,0, 0,0,0, 0,0,0])
            # y_refN = np.array([x_init[0], x_init[1], x_init[2], 1,0,0,0, 0,0,0, 0,0,0, 0,0,0])
            self.acados_ocp_solver[use_model].set(self.N, 'yref', y_refN)

        # Solve OCP
        self.acados_ocp_solver[use_model].solve()

        # Get u
        w_opt_acados = np.ndarray((self.N, 4))
        x_opt_acados = np.ndarray((self.N + 1, len(x_init)))
        x_opt_acados[0, :] = self.acados_ocp_solver[use_model].get(0, "x")
        for i in range(self.N):
            w_opt_acados[i, :] = self.acados_ocp_solver[use_model].get(i, "u")
            x_opt_acados[i + 1, :] = self.acados_ocp_solver[use_model].get(i + 1, "x")

        
        # get cost
        # cost = self.acados_ocp_solver[use_model].get_cost()
        # print("Cost: ", cost)

        w_opt_acados = np.reshape(w_opt_acados, (-1))
        return w_opt_acados if not return_x else (w_opt_acados, x_opt_acados)