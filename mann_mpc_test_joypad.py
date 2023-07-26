
from idyntree.visualize import MeshcatVisualizer
import idyntree.bindings as idyn
from pathlib import Path
import numpy as np
import manifpy as manif
import yarp

import bipedal_locomotion_framework as blf
import datetime

robot_model_path = "/home/gromualdi/robot-code/robotology-superbuild/build/install/share/ergoCub/robots/ergoCubGazeboV1/model.urdf"
config_path = Path(__file__).parent / "config" / "config_mann.toml"
joypad_config_path = Path(__file__).parent / "config" / "config_joypad.toml"
mpc_config_path = Path(__file__).parent / "config" / "config_mpc.toml"


img_folder =  Path(__file__).parent / "img"

params = blf.parameters_handler.TomlParametersHandler()
params.set_from_file(str(config_path))

params_joypad = blf.parameters_handler.TomlParametersHandler()
params_joypad.set_from_file(str(joypad_config_path))

params_mpc = blf.parameters_handler.TomlParametersHandler()
params_mpc.set_from_file(str(mpc_config_path))

ml = idyn.ModelLoader()
ml.loadReducedModelFromFile(robot_model_path,params.get_parameter_vector_string("joints_list"))

viz = MeshcatVisualizer()
# viz.load_model(ml.model(), color=0.0)

mann_trajectory_generator = blf.ml.MANNTrajectoryGenerator()
assert mann_trajectory_generator.set_robot_model(ml.model())
assert mann_trajectory_generator.initialize(params)


joint_positions = np.array([-0.10922017141063572, 0.05081325960010118, 0.06581966291990003, -0.0898053099824925, -0.09324922528169599, -0.05110058859172172,
                            -0.11021232812838086, 0.054291515925228385,0.0735575862560208, -0.09509332143185895, -0.09833823347493076, -0.05367281245082792,
                            0.1531558711397399, -0.001030634273454133, 0.0006584764419034815,
                            -0.0016821925351926288, -0.004284529460797688, 0.030389771690123243,
                            -0.040592118429752494, -0.1695472679986807, -0.20799422095574033, 0.045397975984119654,
                            -0.03946672931050908, -0.16795588539580256, -0.20911090583076936, 0.0419854257806720])




kindyn = idyn.KinDynComputations()
kindyn.loadRobotModel(ml.model())
l_sole_index = ml.model().getFrameIndex("l_sole")
r_sole_index = ml.model().getFrameIndex("r_sole")

link_base_index = ml.model().getFrameLink(l_sole_index)
link_frame_name = ml.model().getFrameName(link_base_index)
l_foot_H_l_sole = kindyn.getRelativeTransform(l_sole_index, link_base_index)
kindyn.setFloatingBase(link_frame_name)

dummy_joint_vel = np.zeros(joint_positions.shape[0])
dummy_gravity = np.zeros(3)
kindyn.setRobotState(l_foot_H_l_sole, joint_positions, idyn.Twist.Zero(), dummy_joint_vel, dummy_gravity)

left_foot_pose = blf.conversions.to_manif_pose(kindyn.getWorldTransform(l_sole_index))
base_pose = blf.conversions.to_manif_pose(kindyn.getWorldTransform("root_link"))
right_foot_pose =  blf.conversions.to_manif_pose(kindyn.getWorldTransform(r_sole_index))

left_foot_pose_new_position = left_foot_pose.translation()
left_foot_pose_new_position[0] = -base_pose.translation()[0] + left_foot_pose.translation()[0]
left_foot_pose_new_position[1] = -base_pose.translation()[1] + left_foot_pose.translation()[1]
left_foot_pose_new_position[2] = 0
left_foot_pose = manif.SE3(left_foot_pose_new_position, manif.SO3.Identity().coeffs())

right_foot_pose_new_position = right_foot_pose.translation()
right_foot_pose_new_position[0] = - base_pose.translation()[0] + right_foot_pose.translation()[0]
right_foot_pose_new_position[1] = - base_pose.translation()[1] + right_foot_pose.translation()[1]
right_foot_pose_new_position[2] = 0
right_foot_pose = manif.SE3(right_foot_pose_new_position, manif.SO3.Identity().coeffs())

new_base_position = base_pose.translation()
new_base_position[0] = 0
new_base_position[1] = 0

base_pose = manif.SE3(new_base_position, base_pose.quat())

left_foot = blf.contacts.EstimatedContact()
right_foot = blf.contacts.EstimatedContact()

left_foot.is_active = True
left_foot.name = "left foot"
left_foot.index = ml.model().getFrameIndex("l_sole")
left_foot.switch_time = datetime.timedelta(seconds=0.0)
left_foot.pose = left_foot_pose

right_foot.is_active = True
right_foot.name = "right foot"
right_foot.index = ml.model().getFrameIndex("r_sole")
right_foot.switch_time = datetime.timedelta(seconds=0.0)
right_foot.pose = right_foot_pose

mann_trajectory_generator.set_initial_state(joint_positions, left_foot, right_foot, base_pose, datetime.timedelta(seconds=0.0))

input_builder = blf.ml.MANNAutoregressiveInputBuilder()
assert input_builder.initialize(params_joypad)

input_builder_input = blf.ml.MANNDirectionalInput()

# mpc
mpc = blf.reduced_model_controllers.CentroidalMPC()
assert mpc.initialize(params_mpc)

centroidal_dynamics = blf.continuous_dynamical_system.CentroidalDynamics()
integrator = blf.continuous_dynamical_system.CentroidalDynamicsRK4Integrator()

sampling_time = params_mpc.get_parameter_datetime("sampling_time")

index = 0

integrator.set_integration_step(sampling_time)
assert integrator.set_dynamical_system(centroidal_dynamics)

# Open and connect YARP port to retrieve joystick input
yarp.Network.init()

# # Open and connect YARP port to retrieve joystick input
p_in = yarp.BufferedPortVector()
p_in.open("/joypad/goal:i")
yarp.Network.connect("/joypad/goal:o", "/joypad/goal:i")


input_builder_input.motion_direction = np.array([0, 0])
input_builder_input.facing_direction = np.array([1, 0])

absolute_index = 0

input()

is_first_iteration = True

while(True):

    # Read from the input port
    res = p_in.read(shouldWait=False)
    # if res:
    input_builder_input.motion_direction = np.array([res[0], res[1]])
    input_builder_input.facing_direction = np.array([res[2], res[3]])

    input_builder.set_input(input_builder_input)
    input_builder.advance()

    generator_input = blf.ml.MANNTrajectoryGeneratorInput()
    if is_first_iteration:
        generator_input.merge_point_index = 0
    else:
        generator_input.merge_point_index = 1
    
    generator_input.desired_future_base_trajectory = input_builder.get_output().desired_future_base_trajectory
    generator_input.desired_future_base_velocities = input_builder.get_output().desired_future_base_velocities
    generator_input.desired_future_facing_directions = input_builder.get_output().desired_future_facing_directions

    mann_trajectory_generator.set_input(generator_input)

    tic = blf.clock().now()
    if index == 0:
        if not mann_trajectory_generator.advance():
            blf.log().error("Error in trajectory generation")
            break
    
    index += 1
    index = index % 1

    mann_output = mann_trajectory_generator.get_output()
    

    # MPC
    if is_first_iteration:
        centroidal_dynamics.set_state((mann_output.com_trajectory[0], 
                                       [0,0,0], [0,0,0]))
        for i in range(len(mann_output.com_trajectory)):
            viz.load_sphere(0.01, "com_adherent_{}".format(i), [69/255, 181/255, 79/255, (1 - i / (len(mann_output.com_trajectory) - 1))])

    comp, comv, angular_momentum = centroidal_dynamics.get_state()
    tmp = mann_output.angular_momentum_trajectory

    for i in range(len(tmp)):
        tmp[i] = tmp[i]  / 56

    assert mpc.set_reference_trajectory(mann_output.com_trajectory, tmp)
    assert mpc.set_state(comp, comv, angular_momentum)
    assert mpc.set_contact_phase_list(mann_output.phase_list)
    
    assert mpc.advance()
    centroidal_dynamics.set_control_input((mpc.get_output().contacts, [0] * 6))
    integrator.integrate(datetime.timedelta(seconds=0.0), sampling_time)

    toc = blf.clock().now()

    blf.log().info("Time elapsed: {}".format(toc - tic))

    if is_first_iteration:
        for i in range(len(mpc.get_output().com_trajectory)):
            viz.load_sphere(0.01, "com_mpc_{}".format(i), [240 / 255, 71 /255, 7/255, (1 - i / (len(mpc.get_output().com_trajectory) - 1))])


    # print contact phase list
    # for phase in mann_output.phase_list:
    #     blf.log().warn("Init time: {}, end time {}".format(phase.begin_time, phase.end_time)) 
    #     for name, contact in phase.active_contacts.items():
    #         blf.log().info("{}".format(contact))

    viz.set_primitive_geometry_transform(comp, np.eye(3), "com")
    for i in range(len(mann_output.com_trajectory)):
        viz.set_primitive_geometry_transform(mann_output.com_trajectory[i], np.eye(3), "com_adherent_{}".format(i))

    for i in range(len(mpc.get_output().com_trajectory)):
        viz.set_primitive_geometry_transform(mpc.get_output().com_trajectory[i], np.eye(3), "com_mpc_{}".format(i))
    

    if is_first_iteration:
        for key, contact in mpc.get_output().contacts.items():
            i = 0
            for corner in contact.corners:
                viz.load_arrow(0.01, "{}_corner_{}".format(key, i), [36/255, 128/255, 218/255, 0.8])
                i += 1
            

    for key, contact in mpc.get_output().contacts.items():
        i = 0
        for corner in contact.corners:
            absolute_corner_position = contact.pose.act(corner.position)
            viz.set_arrow_transform(absolute_corner_position, corner.force / 2, "{}_corner_{}".format(key, i))
            i += 1




    # for i in range(len(mann_output.joint_positions)):
    #     # blf.clock().sleep_for(datetime.timedelta(seconds=0.1))
    i = 0
    """ viz.set_multibody_system_state(mann_output.base_pose[i].translation(),
                                    mann_output.base_pose[i].rotation(),
                                    mann_output.joint_positions[i]) """


    viz.viewer.get_image().save(str(img_folder / "img_{}.png".format(absolute_index)))
    absolute_index += 1

    is_first_iteration = False
    
    # input()

#    blf.clock().sleep_for(datetime.timedelta(seconds=0.01)) """