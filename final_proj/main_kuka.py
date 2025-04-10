import pybullet as p
import pybullet_data
import time
import numpy as np
from attrdict import AttrDict


# Connect to PyBullet and set up the environment
p.connect(p.GUI)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Hides the PyBullet GUI

# Load the KUKA robot and environment objects
planeId = p.loadURDF("plane.urdf")
cuboid_green_id = p.loadURDF("./object/block.urdf", [0.54, 0, 0.02], [0, 0, 0, 1])
kuka_id = p.loadURDF("kuka_iiwa/kuka_with_prismatic_gripper.urdf")

p.setGravity(0, 0, -10)
p.resetDebugVisualizerCamera( # Set the camera view
    cameraDistance=2.0,    # Zoom level (2 meters away)
    cameraYaw=75,          # 45 degrees rotation around the vertical axis
    cameraPitch=-40,       # Tilt 30 degrees downward
    cameraTargetPosition=[0, 0, 0]  # Focus on the origin (0, 0, 0)
)

sim_time_step = 1 / 240  # TODO
eff_index = 7

numJoints = p.getNumJoints(kuka_id)  # kuka + gripper

# Initialize the joints dictionary
joints = AttrDict()

# Populate the joints dictionary with information about each joint
for joint_index in range(p.getNumJoints(kuka_id)):
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
        # print(quaternion)
        # quaternion = (0,1,0,1)
        lower_limits = [-np.pi]*6
        upper_limits = [np.pi]*6
        joint_ranges = [2*np.pi]*6
        rest_poses = [(-0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0)] # rest pose of our ur5 robot

        joint_angles = p.calculateInverseKinematics(
            kuka_id, eff_index, position, quaternion, 
            jointDamping=[0.01]*10 , upperLimits=upper_limits, 
            lowerLimits=lower_limits, jointRanges=joint_ranges, 
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
        # print("tilt: ", get_object_state(cuboid_green_id)[1])
        p.stepSimulation()
        time.sleep(sim_time_step)  # Match real-time simulatio


def position_path(t, t_max, start_pos, end_pos):

    return start_pos + (end_pos - start_pos) * (t / t_max)


def orientation_path(t, t_max, start_orient, end_orient):
    """ orientation is in Euler. """

    return start_orient + (end_orient - start_orient) * (t / t_max)


def get_current_eff_pose():
    linkstate = p.getLinkState(kuka_id, eff_index, computeForwardKinematics=True)
    position, orientation = linkstate[0], linkstate[1]
    return (position, p.getEulerFromQuaternion(orientation))


def get_current_joint_angles(kuka_or_gripper=None):
    "revolute joints: kuka + gripper"
    j = p.getJointStates(kuka_id, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    joints = [i[0] for i in j]

    if kuka_or_gripper == 'kuka':
        return joints[:7]  # [0, 1, 2, 3, 4, 5, 6]
    
    elif kuka_or_gripper == 'gripper':
       return joints[8:]  # [8, 9]

    else:
        return joints


def execute_task_space_trajectory(start_pos, final_pos, duration=1):

    all_joint_angles = calculate_ik(final_pos[0], final_pos[1])
    des_kuka_joint_angles = all_joint_angles[:7]  # Use the first 6 joints for UR5
    set_kuka_joint_angles(get_current_joint_angles("kuka"), des_kuka_joint_angles, duration)



def execute_gripper(init_pos, fin_pos, duration=1):
    """
    Smoothly open or close the gripper by interpolating the gripper opening angle.

    Args:
        time_step (float): Time increment for each simulation step.
        duration (float): Total duration for the gripper motion (in seconds).
        init_pos (float): Initial gripper opening distance.
        fin_pos (float): Final gripper opening distance.
    """
    control_joints = ["left_finger_sliding_joint", "right_finger_sliding_joint"]

    poses = []
    indexes = []
    forces = []

    init_pos = np.array([-init_pos, init_pos])
    fin_pos = np.array([-fin_pos, fin_pos])

    for i, name in enumerate(control_joints):
        joint = joints[name]
        poses.append(fin_pos[i])
        indexes.append(joint.id)
        forces.append(joint.maxForce)

    trajectory = interpolate_trajectory(init_pos, fin_pos, duration)

    for q_t in trajectory:
        p.setJointMotorControlArray(
            kuka_id, indexes,
            controlMode=p.POSITION_CONTROL,
            targetPositions=q_t,
            forces=forces
        )
        # print("tilt: ", get_object_state(cuboid_green_id)[1])
        p.stepSimulation()
        time.sleep(sim_time_step)  # Match real-time simulatio


def get_object_state(object_id):

    position, orientation = p.getBasePositionAndOrientation(object_id)
    orientation_euler = p.getEulerFromQuaternion(orientation)
    linear_velocity, angular_velocity = p.getBaseVelocity(object_id)

    return orientation_euler


def interpolate_trajectory(q_start, q_end, duration):
    num_steps = int(duration / sim_time_step)
    trajectory = []
    for t in range(num_steps):
        alpha = t / (num_steps - 1)  # Normalized time [0, 1]
        q_t = [q_start[i] + alpha * (q_end[i] - q_start[i]) for i in range(len(q_start))]
        trajectory.append(q_t)

    return trajectory


def execute_pick_and_place(third_pos, fourth_pos):

        execute_task_space_trajectory(get_current_eff_pose(), third_pos, duration=1)
        execute_gripper(init_pos=0.01, fin_pos=.00085, duration=.5)  # close gripper
        execute_task_space_trajectory(third_pos, fourth_pos, duration=1)
        tilt_angle = get_object_state(cuboid_green_id)[1]
        execute_task_space_trajectory(fourth_pos, third_pos, duration=1)
        execute_gripper(init_pos=0., fin_pos=.01, duration=.5)  # open gripper
        execute_task_space_trajectory(third_pos, fourth_pos, duration=1)

        return tilt_angle


def main():

    tilt_angle = .55 # tehta: .001

    ## generalized position of end-effector: position + orientation (Euler)
    third_pos = np.array([[tilt_angle, 0., 0.08], [-np.pi/2, 0., -np.pi/2]])  # right on top of the object, ready to grip the object
    fourth_pos = np.array([[tilt_angle, 0., 0.2], [-np.pi/2, 0., -np.pi/2]])  # right on top of the object

    init_kuka_joint_angle = np.array([0.]*7)
    des_kuka_joint_angle = np.array([0., 0.114, 0., -1.895, 0., 1.13, 0.])  # on top of the object with distance in z-axis

    ## initial configuration:
    set_kuka_joint_angles(init_kuka_joint_angle, des_kuka_joint_angle, duration=2)
    execute_gripper(init_pos=0., fin_pos=.01, duration=1)  # open gripper

    execute_pick_and_place(third_pos, fourth_pos)
    

    while True:
        p.stepSimulation()
        time.sleep(sim_time_step)
        pass


if __name__ == '__main__':
    main()