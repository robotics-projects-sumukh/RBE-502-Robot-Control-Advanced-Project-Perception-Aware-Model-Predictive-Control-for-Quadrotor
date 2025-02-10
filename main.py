import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

from quadrotor import Quadrotor3D
from controller import Controller

from utils import q_dot_q, quaternion_inverse, v_dot_q

def trackTrajectory():
    dt = 0.05   # Time step
    N = 20      # Horizontal length
    
    quad = Quadrotor3D()    # Quadrotor model
    controller = Controller(
        quad, 
        q_cost=np.array([10, 10, 10, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01]),
        r_cost=np.array([0.1, 0.01, 0.01, 0.01]),
        t_horizon=2*N*dt,
        n_nodes=N
    )  # Initialize MPC controller

    # Simulation time
    sim_time = 20  # seconds

    # Start and goal positions for the quadrotor and the target object location in the world frame
    start = np.array([0,0,5])
    goal = np.array([10,0,5])
    target_object = np.array([5, 5, 5])

    # Initial position, velocity, orientation, and angular velocity
    quad.pos = start.copy()
    quad.vel = np.array([0,0,0])
    # quad.angle = np.array([1,0,0,0]) # roll, pitch, yaw as 0 degrees
    quad.angle = np.array([0.9238795, 0.0, 0, 0.3826834]) # yaw as 45 degrees

    quad.Wpf = target_object # position of object in world frame
    quad.pBC = np.array([0, 0, 0]) # position of camera in body frame
    # quad.qBC = np.array([1, 0, 0, 0]) # orientation of camera in body frame
    quad.qBC = np.array([0.7071068, 0.0, 0.7071068, 0.0]) # orientation of camera in body frame, z axis of camera is pointing to the front
    quad.Cpf = np.array([0, 0, 0]) # position of object in camera frame
    quad.Cp_dot_f = np.array([0, 0, 0]) # velocity of object in camera frame

    quad.fx = 0.016 * 640 # focal length in x = 16mm (https://www.canakit.com/raspberry-pi-global-shutter-camera-kit.html)
    quad.fy = 0.016 * 640 # focal length in y = 16mm

    Cpf = v_dot_q(quad.Wpf - (v_dot_q(quad.pBC, quad.angle) + quad.pos), quaternion_inverse(q_dot_q(quad.angle, quad.qBC)))
    print("Initial Cpf: ", Cpf)

    s_initial = np.zeros((3,))
    s_initial[0] = quad.fx * Cpf[0] / Cpf[2]
    s_initial[1] = quad.fy * Cpf[1] / Cpf[2]
    s_initial[2] = 0

    print("Initial projection: ", s_initial)

    quad.s_quad = s_initial
    quad.s_dot_quad = np.zeros((3,))

    # Create a reference trajectory
    xref, yref, zref = createTrajectory(start, goal, sim_time, dt)

    # Initialize lists to store the path, position, orientation, linear velocity, and angular velocity
    path = []
    position = []
    orientation = []
    linear_velocity = []
    angular_velocity = []
    poi_projection = []
    poi_projection_rate = []
    thrust = []

    # Main loop
    time_record = []
    
    # Summary:
    # 1. Get the reference trajectory for the next N steps
    # 2. Run the optimization to get the optimal control input
    # 3. Update the quadrotor state using the control input
    for i in range(int(sim_time/dt)):
        # print(i)
        x = xref[i:i+N+1]; y = yref[i:i+N+1]; z = zref[i:i+N+1]
        if len(x) < N+1:
            x = np.concatenate((x,np.ones(N+1-len(x))*xref[-1]),axis=None)
            y = np.concatenate((y,np.ones(N+1-len(y))*yref[-1]),axis=None)
            z = np.concatenate((z,np.ones(N+1-len(z))*zref[-1]),axis=None)

        # Checkpoints for the next N steps
        local_goal=np.array([x,y,z]).T 
        # print(local_goal)

        # Run the optimization to get the optimal control input
        current = np.concatenate([quad.pos, quad.angle, quad.vel, quad.s_quad, quad.s_dot_quad])
        start_time = timeit.default_timer()
        control_input = controller.run_optimization(initial_state=current, goal=local_goal, mode='traj')[:4]
        time_record.append(timeit.default_timer() - start_time)

        # Update the quadrotor state using the control input
        quad.update(control_input, dt)

        # Store the current state
        path.append(quad.pos)
        position.append(quad.pos)
        orientation.append(quad.angle)
        linear_velocity.append(quad.vel)
        angular_velocity.append(control_input[1:])
        thrust.append(control_input[0])
        poi_projection.append(quad.s_quad)
        poi_projection_rate.append(quad.s_dot_quad)

    # Convert lists to NumPy arrays
    path = np.array(path)
    position = np.array(position)
    orientation = np.array(orientation)
    linear_velocity = np.array(linear_velocity)
    angular_velocity = np.array(angular_velocity)
    poi_projection = np.array(poi_projection)
    poi_projection_rate = np.array(poi_projection_rate)
    thrust = np.array(thrust)
    # Convert quaternions to Euler angles
    euler_angles = quaternion_to_euler(orientation)

    print("Final Cpf: ", quad.Cpf)

    print("Final projection: ", quad.s_quad)

    # CPU time
    print("average estimation time is {:.5f}".format(np.array(time_record).mean()))
    print("max estimation time is {:.5f}".format(np.array(time_record).max()))
    print("min estimation time is {:.5f}".format(np.array(time_record).min()))

    # Visualization
    print("Start: ", path[0])
    print("Goal: ", path[-1])

    # Create a figure with a 3x3 grid
    fig = plt.figure(figsize=(12, 10))

    # Plot the position components (x, y, z) in the first subplot
    ax1 = fig.add_subplot(4, 4, 1)
    ax1.plot(position[:, 0], label='x')
    ax1.plot(position[:, 1], label='y')
    ax1.plot(position[:, 2], label='z')
    ax1.set_title('Position')
    ax1.legend()
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Position [m]')
    ax1.grid(True) 

    # Plot the orientation components (quaternion: w, x, y, z) in the second subplot
    ax2 = fig.add_subplot(4, 4, 2)
    # ax2.plot(orientation[:, 0], label='qw')
    # ax2.plot(orientation[:, 1], label='qx')
    # ax2.plot(orientation[:, 2], label='qy')
    # ax2.plot(orientation[:, 3], label='qz')
    ax2.plot(euler_angles[:, 0], label='roll')
    ax2.plot(euler_angles[:, 1], label='pitch')
    ax2.plot(euler_angles[:, 2], label='yaw')
    ax2.set_title('Orientation (roll, pitch, yaw)')
    ax2.legend()
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Quaternion components')
    ax2.grid(True)

    # Plot the projection of the Point of Interest rate (s1_dot, s2_dot, s3_dot) in the 3rd subplot
    ax3 = fig.add_subplot(4, 4, 3)
    ax3.plot(poi_projection_rate[:, 0], label='s1_dot')
    ax3.plot(poi_projection_rate[:, 1], label='s2_dot')
    ax3.plot(poi_projection_rate[:, 2], label='s3_dot')
    ax3.set_title('PoI Projection Rate')
    ax3.legend()
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Projection Rate')
    ax3.grid(True)
    # ax3.plot(euler_angles[:, 0], label='roll')
    # ax3.plot(euler_angles[:, 1], label='pitch')
    # ax3.plot(euler_angles[:, 2], label='yaw')
    # ax3.set_title('Orientation (roll, pitch, yaw)')
    # ax3.legend()
    # ax3.set_xlabel('Time step')
    # ax3.set_ylabel('Euler angles [rad]')
    # ax3.grid(True)

    # Plot the thrust in the fourth subplot
    ax4 = fig.add_subplot(4, 4, 4)
    ax4.plot(thrust[:], label='thrust')
    ax4.set_title('Thrust')
    ax4.legend()
    ax4.set_xlabel('Time step')
    ax4.set_ylabel('Thrust')
    ax4.grid(True)

    # Plot the projection of the Point of Interest (s1, s2, s3) in the 5th subplot
    ax5 = fig.add_subplot(4, 4, 5)
    ax5.plot(poi_projection[:, 0], label='s1')
    ax5.plot(poi_projection[:, 1], label='s2')
    ax5.plot(poi_projection[:, 2], label='s3')
    ax5.set_title('PoI Projection')
    ax5.legend()
    ax5.set_xlabel('Time step')
    ax5.set_ylabel('Projection [m]')
    ax5.grid(True)

    # Plot the linear velocity components (vx, vy, vz) in the 9th subplot
    ax6 = fig.add_subplot(4, 4, 9)
    ax6.plot(linear_velocity[:, 0], label='vx')
    ax6.plot(linear_velocity[:, 1], label='vy')
    ax6.plot(linear_velocity[:, 2], label='vz')
    ax6.set_title('Linear Velocity')
    ax6.legend()
    ax6.set_xlabel('Time step')
    ax6.set_ylabel('Velocity [m/s]')
    ax6.grid(True)

    # Plot the angular velocity components (wx, wy, wz) in the fourth subplot
    ax7 = fig.add_subplot(4, 4, 13)
    ax7.plot(angular_velocity[:, 0], label='wx')
    ax7.plot(angular_velocity[:, 1], label='wy')
    ax7.plot(angular_velocity[:, 2], label='wz')
    ax7.set_title('Angular Velocity')
    ax7.legend()
    ax7.set_xlabel('Time step')
    ax7.set_ylabel('Angular velocity [rad/s]')
    ax7.grid(True)

    # Create a 3D plot for the trajectory and projections
    ax8 = fig.add_subplot(4, 4, (6, 16), projection='3d')

    # Plot the 3D trajectory
    ax8.plot(xref, yref, zref, c='r', label='Reference Trajectory', linewidth=2)
    ax8.plot(path[:, 0], path[:, 1], path[:, 2], label='Trajectory', linewidth=2)
    ax8.scatter(start[0], start[1], start[2], c='green', label='Start')
    ax8.scatter(goal[0], goal[1], goal[2], c='blue', label='Goal')
    ax8.scatter(target_object[0], target_object[1], target_object[2], c='orange', label='Object')
    ax8.set_xlabel('x [m]')
    ax8.set_ylabel('y [m]')
    ax8.set_zlabel('z [m]')
    ax8.set_title('3D Trajectory')
    ax8.legend()


    # Adjust layout for better spacing
    plt.tight_layout()

    # plt.figure()
    # plt.plot(time_record)
    # plt.legend()
    # plt.ylabel('CPU Time [s]')
    # plt.yscale("log")

    # Show the entire figure with subplots and 3D graph
    plt.show()

def quaternion_to_euler(quaternions):
    """
    Convert an array of quaternions to Euler angles (roll, pitch, yaw).
    
    Parameters:
    quaternions (np.ndarray): An array of shape (N, 4) where N is the number of quaternions,
                              and each quaternion is represented as [qw, qx, qy, qz].

    Returns:
    np.ndarray: An array of shape (N, 3) containing the corresponding Euler angles
                [roll, pitch, yaw] in radians.
    """
    euler_angles = np.zeros((quaternions.shape[0], 3))

    for i in range(quaternions.shape[0]):
        qw, qx, qy, qz = quaternions[i]

        # Calculate roll (phi)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx**2 + qy**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Calculate pitch (theta)
        sinp = 2 * (qw * qy - qz * qx)
        if np.abs(sinp) >= 1:
            pitch = np.sign(sinp) * (np.pi / 2)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Calculate yaw (psi)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy**2 + qz**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Store the angles in radians
        euler_angles[i] = [roll, pitch, yaw]

    return euler_angles

def createTrajectory(start, goal, sim_time, dt):

    xA = np.array([
        [1, 0, 0, 0],
        [1, sim_time**3, sim_time**4, sim_time**5],
        [0, 3*sim_time**2, 4*sim_time**3, 5*sim_time**4],
        [0, 6*sim_time, 12*sim_time**2, 20*sim_time**3]
    ])
    xb = np.array([start[0], goal[0], 0, 0])
    x_coeff = np.linalg.solve(xA, xb)

    yA = np.array([
        [1, 0, 0, 0],
        [1, sim_time**3, sim_time**4, sim_time**5],
        [0, 3*sim_time**2, 4*sim_time**3, 5*sim_time**4],
        [0, 6*sim_time, 12*sim_time**2, 20*sim_time**3]
    ])
    yb = np.array([start[1], goal[1], 0, 0])
    y_coeff = np.linalg.solve(yA, yb)

    zA = np.array([
        [1, 0, 0, 0],
        [1, sim_time**3, sim_time**4, sim_time**5],
        [0, 3*sim_time**2, 4*sim_time**3, 5*sim_time**4],
        [0, 6*sim_time, 12*sim_time**2, 20*sim_time**3]
    ])
    zb = np.array([start[2], goal[2], 0, 0])
    z_coeff = np.linalg.solve(zA, zb)

    xref = [] 
    yref = []
    zref = []
    for i in range(int(sim_time/dt)):
        x = x_coeff[0] + x_coeff[1]*(dt*i)**3 + x_coeff[2]*(dt*i)**4 + x_coeff[3]*(dt*i)**5
        y = y_coeff[0] + y_coeff[1]*(dt*i)**3 + y_coeff[2]*(dt*i)**4 + y_coeff[3]*(dt*i)**5
        z = z_coeff[0] + z_coeff[1]*(dt*i)**3 + z_coeff[2]*(dt*i)**4 + z_coeff[3]*(dt*i)**5
        xref.append(x)
        yref.append(y)
        zref.append(z)
    return np.array(xref), np.array(yref), np.array(zref)

if __name__ == "__main__":
    trackTrajectory()