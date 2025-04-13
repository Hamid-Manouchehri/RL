import pybullet as p
import pybullet_data
import time
import numpy as np
from attrdict import AttrDict

# Connect to PyBullet and set up the environment
p.connect(p.GUI)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Optionally hide the PyBullet GUI

# Load the KUKA robot and environment objects
planeId = p.loadURDF("plane.urdf")
cuboid_green_id = p.loadURDF("./object/block.urdf", [0.54, 0, 0.02], [0, 0, 0, 1])
kuka_id = p.loadURDF("kuka_iiwa/kuka_with_prismatic_gripper.urdf")

p.setGravity(0, 0, -10)
p.resetDebugVisualizerCamera(
    cameraDistance=2.0,    # Zoom level (2 meters away)
    cameraYaw=75,          # Rotation around the vertical axis
    cameraPitch=-40,       # Tilt downward
    cameraTargetPosition=[0, 0, 0]  # Focus on the origin
)

sim_time_step = 1 / 240  # Simulation time step
eff_index = 7           # End-effector link index

numJoints = p.getNumJoints(kuka_id)  # Total joints: KUKA + gripper

# Initialize the joints dictionary
joints = AttrDict()

# Populate the joints dictionary with information about each joint
for joint_index in range(numJoints):
    joint_info = p.getJointInfo(kuka_id, joint_index)
    joint_name = joint_info[1].decode("utf-8")
    joints[joint_name] = AttrDict({
        "id": joint_info[0],
        "lowerLimit": joint_info[8],
        "upperLimit": joint_info[9],
        "maxForce": joint_info[10],
        "maxVelocity": joint_info[11],
    })

def calculate_ik(position, orientation):
    quaternion = p.getQuaternionFromEuler(orientation)
    lower_limits = [-np.pi] * 7
    upper_limits = [np.pi] * 7
    joint_ranges = [2 * np.pi] * 7
    rest_poses = [(-0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0)]  # Rest pose for 7 DOF
    # Calculate inverse kinematics with a 7-element damping vector
    joint_angles = p.calculateInverseKinematics(
        kuka_id, eff_index, position, quaternion,
        jointDamping=[0.01] * 7,
        lowerLimits=lower_limits,
        upperLimits=upper_limits,
        jointRanges=joint_ranges,
        restPoses=rest_poses
    )
    return joint_angles

def set_kuka_joint_angles(init_joint_angles, des_joint_angles, duration):
    control_joints = ["lbr_iiwa_joint_1", "lbr_iiwa_joint_2", "lbr_iiwa_joint_3",
                      "lbr_iiwa_joint_4", "lbr_iiwa_joint_5", "lbr_iiwa_joint_6",
                      "lbr_iiwa_joint_7"]
    poses = []
    indexes = []
    forces = []
    for i, name in enumerate(control_joints):
        joint = joints[name]
        poses.append(des_joint_angles[i])
        indexes.append(joint.id)
        forces.append(joint.maxForce)
    trajectory = interpolate_trajectory(init_joint_angles, des_joint_angles, duration)
    for q_t in trajectory:
        p.setJointMotorControlArray(
            kuka_id, indexes,
            controlMode=p.POSITION_CONTROL,
            targetPositions=q_t,
            forces=forces
        )
        p.stepSimulation()
        time.sleep(sim_time_step)

def position_path(t, t_max, start_pos, end_pos):
    return start_pos + (end_pos - start_pos) * (t / t_max)

def orientation_path(t, t_max, start_orient, end_orient):
    """Orientation path (Euler angles)."""
    return start_orient + (end_orient - start_orient) * (t / t_max)

def get_current_eff_pose():
    linkstate = p.getLinkState(kuka_id, eff_index, computeForwardKinematics=True)
    position, orientation = linkstate[0], linkstate[1]
    position = list(position)
    position[2] = position[2] - 0.0491
    return (position, list(p.getEulerFromQuaternion(orientation)))

def get_current_joint_angles(kuka_or_gripper=None):
    joint_states = p.getJointStates(kuka_id, list(range(numJoints)))
    joint_values = [state[0] for state in joint_states]
    if kuka_or_gripper == 'kuka':
        return joint_values[:7]  # First 7 joints for the KUKA arm
    elif kuka_or_gripper == 'gripper':
        return joint_values[7:]  # Remaining joints for the gripper
    else:
        return joint_values

def interpolate_trajectory(q_start, q_end, duration):
    num_steps = int(duration / sim_time_step) + 1
    trajectory = []
    for t in range(num_steps):
        alpha = t / (num_steps - 1)
        q_t = [q_start[i] + alpha * (q_end[i] - q_start[i]) for i in range(len(q_start))]
        trajectory.append(q_t)
    return trajectory

###############################################################################
# New function: moveL (linear Cartesian motion)
def moveL(robot_id, eff_idx, start_pos, end_pos, target_ori, duration, time_step=sim_time_step):
    """
    Moves the robot's end-effector in a straight line from start_pos to end_pos.
    
    Parameters:
      robot_id      : Unique ID of the robot in PyBullet.
      eff_idx       : End-effector link index.
      start_pos     : Starting Cartesian position [x, y, z].
      end_pos       : Ending Cartesian position [x, y, z].
      target_ori    : Desired constant end-effector orientation (Euler angles).
      duration      : Movement duration in seconds.
      time_step     : Time step to update simulation.
    """
    num_steps = int(duration / time_step) + 1
    target_quat = p.getQuaternionFromEuler(target_ori)
    for step in range(num_steps):
        alpha = step / (num_steps - 1)
        current_pos = np.array(start_pos) + alpha * (np.array(end_pos) - np.array(start_pos))
        # Calculate inverse kinematics for the current target position and fixed orientation
        joint_poses = p.calculateInverseKinematics(robot_id, eff_idx, current_pos.tolist(), target_quat)
        # Assuming that the arm joints are the first 7 joints
        control_joint_indices = list(range(7))
        target_positions = joint_poses[:7]
        p.setJointMotorControlArray(robot_id, control_joint_indices,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=target_positions)
        p.stepSimulation()
        time.sleep(time_step)
###############################################################################

def execute_task_space_trajectory(start_pos, final_pos, duration=1):
    # final_pos is a two-element structure: [position, orientation]
    all_joint_angles = calculate_ik(final_pos[0], final_pos[1])
    des_kuka_joint_angles = all_joint_angles[:7]  # Use the first 7 joint angles
    set_kuka_joint_angles(get_current_joint_angles("kuka"), des_kuka_joint_angles, duration)

def execute_gripper(init_pos, fin_pos, duration=1):
    """
    Smoothly open or close the gripper by interpolating the gripper opening angle.
    Args:
        duration (float): Duration for the gripper motion (seconds).
        init_pos (float): Initial gripper opening distance.
        fin_pos (float): Final gripper opening distance.
    """
    control_joints = ["left_finger_sliding_joint", "right_finger_sliding_joint"]
    poses = []
    indexes = []
    forces = []
    
    init_arr = np.array([-init_pos, init_pos])
    fin_arr = np.array([-fin_pos, fin_pos])
    for i, name in enumerate(control_joints):
        joint = joints[name]
        poses.append(fin_arr[i])
        indexes.append(joint.id)
        forces.append(joint.maxForce)
    trajectory = interpolate_trajectory(init_arr, fin_arr, duration)
    for q_t in trajectory:
        p.setJointMotorControlArray(
            kuka_id, indexes,
            controlMode=p.POSITION_CONTROL,
            targetPositions=q_t,
            forces=forces
        )
        p.stepSimulation()
        time.sleep(sim_time_step)

def get_object_state(object_id):
    position, orientation = p.getBasePositionAndOrientation(object_id)
    orientation_euler = p.getEulerFromQuaternion(orientation)
    return orientation_euler


def main():

    init_kuka_joint_angle = np.array([0.0] * 7)
    des_kuka_joint_angle = np.array([-0., 0.44, 0., -2.086, -0., 0.615, -0.])
    
    # Set the initial configuration of KUKA and open gripper
    set_kuka_joint_angles(init_kuka_joint_angle, des_kuka_joint_angle, duration=2)
    execute_gripper(init_pos=0., fin_pos=0.01, duration=1)  # Open gripper

    print(get_current_eff_pose(),'\n')

    moveL(kuka_id, eff_index, [0.41, 0., 0.2025], [0.41, 0., 0.08], [-np.pi/2, 0., -np.pi/2], duration=1, time_step=sim_time_step)
    execute_gripper(init_pos=0.01, fin_pos=0.0008, duration=0.5)  # Close gripper
    moveL(kuka_id, eff_index, [0.41, 0., 0.08], [0.41, 0., 0.2025], [-np.pi/2, 0., -np.pi/2], duration=1, time_step=sim_time_step)
    moveL(kuka_id, eff_index, [0.41, 0., 0.2025], [0.41, 0., 0.08], [-np.pi/2, 0., -np.pi/2], duration=1, time_step=sim_time_step)
    execute_gripper(init_pos=0., fin_pos=0.01, duration=1)  # Open gripper
    moveL(kuka_id, eff_index, [0.41, 0., 0.08], [0.41, 0., 0.2025], [-np.pi/2, 0., -np.pi/2], duration=1, time_step=sim_time_step)

    print(get_current_eff_pose(),'\n')

    moveL(kuka_id, eff_index, [0.41, 0., 0.2025], [0.47, 0., 0.2025], [-np.pi/2, 0., -np.pi/2], duration=1, time_step=sim_time_step)

    print(get_current_eff_pose(),'\n')

    moveL(kuka_id, eff_index, [0.47, 0., 0.2025], [0.47, 0., 0.08], [-np.pi/2, 0., -np.pi/2], duration=1, time_step=sim_time_step)
    execute_gripper(init_pos=0.01, fin_pos=0.0008, duration=0.5)  # Close gripper
    moveL(kuka_id, eff_index, [0.47, 0., 0.08], [0.47, 0., 0.2025], [-np.pi/2, 0., -np.pi/2], duration=1, time_step=sim_time_step)
    moveL(kuka_id, eff_index, [0.47, 0., 0.2025], [0.47, 0., 0.08], [-np.pi/2, 0., -np.pi/2], duration=1, time_step=sim_time_step)
    execute_gripper(init_pos=0., fin_pos=0.01, duration=1)  # Open gripper
    moveL(kuka_id, eff_index, [0.47, 0., 0.08], [0.47, 0., 0.2025], [-np.pi/2, 0., -np.pi/2], duration=1, time_step=sim_time_step)
    
    print(get_current_eff_pose(),'\n')

    # Continue simulation indefinitely to observe the final configuration
    while True:
        p.stepSimulation()
        time.sleep(sim_time_step)

if __name__ == '__main__':
    main()
