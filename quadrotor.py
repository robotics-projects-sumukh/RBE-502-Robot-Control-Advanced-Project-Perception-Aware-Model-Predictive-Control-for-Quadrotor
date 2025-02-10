from math import sqrt
import numpy as np
from utils import quaternion_to_euler, skew_symmetric, v_dot_q, unit_quat, quaternion_inverse, q_dot_q


class Quadrotor3D:

    def __init__(self, noisy=False, drag=False, payload=False, motor_noise=False):
        """
        Initialization of the 3D quadrotor class
        :param noisy: Whether noise is used in the simulation
        :type noisy: bool
        :param drag: Whether to simulate drag or not.
        :type drag: bool
        :param payload: Whether to simulate a payload force in the simulation
        :type payload: bool
        :param motor_noise: Whether non-gaussian noise is considered in the motor inputs
        :type motor_noise: bool
        """

        # Either 'x' or '+'. The xy configuration of the thrusters in the body plane.
        configuration = 'x'

        # Maximum thrust in Newtons of a thruster when rotating at maximum speed.
        self.max_thrust = 20

        # System state space
        self.pos = np.zeros((3,))
        self.vel = np.zeros((3,))
        self.angle = np.array([1., 0., 0., 0.])  # Quaternion format: qw, qx, qy, qz

        # control input
        # self.a_rate = np.zeros((3,))

        # target object data
        self.s_quad = np.zeros((3,))
        self.s_dot_quad = np.zeros((3,))

        self.Wpf = np.zeros((3,)) # position of object in world frame
        self.pBC = np.zeros((3,)) # position of camera in body frame
        # self.qBC = np.array([1., 0., 0., 0.]) # orientation of camera in body frame
        self.qBC = np.array([0.7071068, 0.0, 0.7071068, 0.0]) 
        self.Cpf = np.zeros((3,)) # position of object in camera frame
        self.Cp_dot_f = np.zeros((3,)) # velocity of object in camera frame

        self.fx = 0.0 # focal length in x
        self.fy = 0.0 # focal length in y

        # Input constraints
        self.max_input_value = 1  # Motors at full thrust
        self.min_input_value = 0  # Motors turned off

        # Quadrotor intrinsic parameters
        self.J = np.array([.03, .03, .06])  # N m s^2 = kg m^2
        self.mass = 0.5  # kg

        # Length of motor to CoG segment
        self.length = 0.38 / 2  # m

        # Positions of thrusters
        if configuration == '+':
            self.x_f = np.array([self.length, 0, -self.length, 0])
            self.y_f = np.array([0, self.length, 0, -self.length])
        elif configuration == 'x':
            h = np.cos(np.pi / 4) * self.length
            self.x_f = np.array([h, -h, -h, h])
            # self.y_f = np.array([-h, -h, h, h])
            self.y_f = np.array([h, h, -h, -h])

        # For z thrust torque calculation
        self.c = 0.013  # m   (z torque generated by each motor)
        self.z_l_tau = np.array([-self.c, self.c, -self.c, self.c])

        # Gravity vector
        self.g = np.array([[0], [0], [9.81]])  # m s^-2

        # Actuation thrusts
        self.u_noiseless = np.array([0.0, 0.0, 0.0, 0.0])
        self.u = np.array([0.0, 0.0, 0.0, 0.0])  # c, omega_x, omega_y, omega_z

        # Drag coefficients [kg / m]
        self.rotor_drag_xy = 0.3
        self.rotor_drag_z = 0.0  # No rotor drag in the z dimension
        self.rotor_drag = np.array([self.rotor_drag_xy, self.rotor_drag_xy, self.rotor_drag_z])[:, np.newaxis]
        self.aero_drag = 0.08

        self.drag = drag
        self.noisy = noisy
        self.motor_noise = motor_noise

        self.payload_mass = 0.3  # kg
        self.payload_mass = self.payload_mass * payload

    def set_state(self, *args, **kwargs):
        if len(args) != 0:
            assert len(args) == 1 and len(args[0]) == 16
            self.pos[0], self.pos[1], self.pos[2], \
            self.angle[0], self.angle[1], self.angle[2], self.angle[3], \
            self.vel[0], self.vel[1], self.vel[2], \
            self.s_quad[0], self.s_quad[1], self.s_quad[2], \
            self.s_dot_quad[0], self.s_dot_quad[1], self.s_dot_quad[2] \
                = args[0]

        else:
            self.pos = kwargs["pos"]
            self.angle = kwargs["angle"]
            self.vel = kwargs["vel"]
            self.s_quad = kwargs["s"]
            self.s_dot_quad = kwargs["s_dot"]

    def get_state(self, quaternion=False, stacked=False):

        if quaternion and not stacked:
            return [self.pos, self.angle, self.vel, self.s_quad, self.s_dot_quad]
        if quaternion and stacked:
            return [self.pos[0], self.pos[1], self.pos[2], 
                    self.angle[0], self.angle[1], self.angle[2], self.angle[3],
                    self.vel[0], self.vel[1], self.vel[2], 
                    self.s_quad[0], self.s_quad[1], self.s_quad[2],
                    self.s_dot_quad[0], self.s_dot_quad[1], self.s_dot_quad[2]]

        angle = quaternion_to_euler(self.angle)
        if not quaternion and stacked:
            return [self.pos[0], self.pos[1], self.pos[2], 
                    angle[0], angle[1], angle[2],
                    self.vel[0], self.vel[1], self.vel[2],
                    self.s_quad[0], self.s_quad[1], self.s_quad[2],
                    self.s_dot_quad[0], self.s_dot_quad[1], self.s_dot_quad[2]]
        
        return [self.pos, angle, self.vel,self.s_quad, self.s_dot_quad]

    def get_control(self, noisy=False):
        if not noisy:
            return self.u_noiseless
        else:
            return self.u

    def update(self, u, dt):
        """
        Runge-Kutta 4th order dynamics integration

        :param u: 4-dimensional vector with components between [0.0, 1.0] that represent the activation of each motor.
        :param dt: time differential
        """
        f_d = np.zeros((3, 1))
        x = self.get_state(quaternion=True, stacked=False)

        # RK4 integration
        k1 = [self.f_pos(x), self.f_att(x, u), self.f_vel(x, u, f_d)]
        x_aux = [x[i] + dt / 2 * k1[i] for i in range(3)]
        k2 = [self.f_pos(x_aux), self.f_att(x_aux, u), self.f_vel(x_aux, u, f_d)]
        x_aux = [x[i] + dt / 2 * k2[i] for i in range(3)]
        k3 = [self.f_pos(x_aux), self.f_att(x_aux, u), self.f_vel(x_aux, u, f_d)]
        x_aux = [x[i] + dt * k3[i] for i in range(3)]
        k4 = [self.f_pos(x_aux), self.f_att(x_aux, u), self.f_vel(x_aux, u, f_d)]
        x = [x[i] + dt * (1.0 / 6.0 * k1[i] + 2.0 / 6.0 * k2[i] + 2.0 / 6.0 * k3[i] + 1.0 / 6.0 * k4[i]) for i in
             range(3)]

        x.append(self.update_s(x, u))
        x.append(self.update_s_dot(x, u))

        # Ensure unit quaternion
        x[1] = unit_quat(x[1])

        self.pos, self.angle, self.vel, self.s_quad, self.s_dot_quad = x

    def f_pos(self, x):
        """
        Time-derivative of the position vector
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :return: position differential increment (vector): d[pos_x; pos_y]/dt
        """

        vel = x[2]
        return vel

    def f_att(self, x, u):
        """
        Time-derivative of the attitude in quaternion form
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :return: attitude differential increment (quaternion qw, qx, qy, qz): da/dt
        """

        rate = u[1:]
        angle_quaternion = x[1]

        return 1 / 2 * skew_symmetric(rate).dot(angle_quaternion)

    def f_vel(self, x, u, f_d):
        """
        Time-derivative of the velocity vector
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :param u: control input vector (4-dimensional): [trust_motor_1, ..., thrust_motor_4]
        :param f_d: disturbance force vector (3-dimensional)
        :return: 3D velocity differential increment (vector): d[vel_x; vel_y; vel_z]/dt
        """

        a_thrust = np.array([[0], [0], [u[0] * self.max_thrust]])

        if self.drag:
            # Transform velocity to body frame
            v_b = v_dot_q(x[2], quaternion_inverse(x[1]))[:, np.newaxis]
            # Compute aerodynamic drag acceleration in world frame
            a_drag = -self.aero_drag * v_b ** 2 * np.sign(v_b) / self.mass
            # Add rotor drag
            a_drag -= self.rotor_drag * v_b / self.mass
            # Transform drag acceleration to world frame
            a_drag = v_dot_q(a_drag, x[1])
        else:
            a_drag = np.zeros((3, 1))

        angle_quaternion = x[1]

        a_payload = -self.payload_mass * self.g / self.mass

        return np.squeeze(-self.g + a_payload + a_drag + v_dot_q(a_thrust + f_d / self.mass, angle_quaternion))
    
    def update_s(self, x, u):
        pos = x[0]
        angle = x[1]

        self.Cpf = v_dot_q(self.Wpf - (v_dot_q(self.pBC, angle) + pos), quaternion_inverse(q_dot_q(angle, self.qBC)))

        s = np.squeeze(np.array(
            [[(self.fx * self.Cpf[0] / self.Cpf[2])],
            [(self.fy * self.Cpf[1] / self.Cpf[2])],
            [0]]))
        
        return s
    
    def update_s_dot(self, x, u):
        angle = x[1]
        vel = x[2]
        rate = u[1:]

        omega_c = v_dot_q(rate, quaternion_inverse(self.qBC))

        skew_omega_B = np.array([[0, -rate[2], rate[1]], 
                                 [rate[2], 0, -rate[0]],
                                 [-rate[1], rate[0], 0]])
        CvWC = v_dot_q((0.5*skew_omega_B.dot(v_dot_q(self.pBC, angle)) + vel), quaternion_inverse(q_dot_q(angle, self.qBC)))

        skew_omega_C = np.array([[0, -omega_c[2], omega_c[1]], 
                                 [omega_c[2], 0, -omega_c[0]], 
                                 [-omega_c[1], omega_c[0], 0]])
        
        self.Cp_dot_f = - 0.5*skew_omega_C.dot(self.Cpf) - CvWC

        s_dot = np.squeeze(np.array(
            [[-self.fx/self.Cpf[2]**2 * (self.Cpf[2]*self.Cp_dot_f[0] - self.Cpf[0]*self.Cp_dot_f[2])],
            [self.fy/self.Cpf[2]**2 * (self.Cpf[1]*self.Cp_dot_f[2] - self.Cpf[2]*self.Cp_dot_f[1])],
            [0]]))

        return s_dot


    