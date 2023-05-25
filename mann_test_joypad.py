
from idyntree.visualize import MeshcatVisualizer
import idyntree.bindings as idyn
from pathlib import Path
import numpy as np
import manifpy as manif
import yarp

import bipedal_locomotion_framework as blf
import datetime

robot_model_path = "/home/gromualdi/robot-install/share/ergoCub/robots/ergoCubGazeboV1/model.urdf"
config_path = Path(__file__).parent / "config" / "config_mann.toml"
joypad_config_path = Path(__file__).parent / "config" / "config_joypad.toml"


params = blf.parameters_handler.TomlParametersHandler()
params.set_from_file(str(config_path))

params_joypad = blf.parameters_handler.TomlParametersHandler()
params_joypad.set_from_file(str(joypad_config_path))


ml = idyn.ModelLoader()
ml.loadReducedModelFromFile(robot_model_path,params.get_parameter_vector_string("joints_list"))

viz = MeshcatVisualizer()
viz.load_model(ml.model())

mann_trajectory_generator = blf.ml.MANNAutoregressive()
mann_trajectory_generator.set_robot_model(ml.model())
mann_trajectory_generator.initialize(params)


# reset the mann with the initial valuereshape(a, newshape, order='C')[source]
mann_input = blf.ml.MANNInput()
mann_input.base_position_trajectory = np.reshape(np.array([-2.5275008578018454e-06, 1.090306280898421e-06, -1.5506335119888637e-06, -3.236287161853386e-07,
-5.71689704839673e-07, -5.110262834502992e-07, -6.774263245148959e-08, -2.761977570107203e-07,
4.2237717419110754e-08, -8.198966694632619e-08, 0.0, 0.0,
-0.00018970073152083794, 2.91193163688084e-05, -0.0017083972855563671, 4.915881001195277e-05,
 -0.0022356689573097416, 0.0002596790684395351, -0.0018592437848407375, -7.129441343432287e-05,
-0.0009642088339483212, -0.00010555693969488232, 0.0, 0.0]), (2, 12), order='F')

mann_input.base_velocity_trajectory = np.reshape(np.array([0.004595956792923565, 0.0006906726560392635, 0.004598776458800517, 0.0006938221584109283,
0.004600030181431486, 0.0006978252906316189, 0.0046003567627922945, 0.0006998395362234365,
0.004600297680422853, 0.0007002354775520248, 0.004600120788443449, 0.0007000091724630841,
-0.005222550612271976, 0.0009446655060324624, -0.005888295415666073, 0.0023566472197978287,
-0.003353321711468564, -0.001953677449355967, -0.0011558440154326999, -0.0016930928367522113,
0.0009471750849568486, 0.0003476954526010457, 0.0, 1.3552527156068805e-20]), (2, 12), order='F')

mann_input.facing_direction_trajectory = np.reshape(np.array([1.0, 0.0] * 12), (2, 12), order='F')

mann_input.joint_positions = np.array([-0.10922017141063572, 0.05081325960010118, 0.06581966291990003, -0.0898053099824925, -0.09324922528169599, -0.05110058859172172,
                                       -0.11021232812838086, 0.054291515925228385,0.0735575862560208, -0.09509332143185895, -0.09833823347493076, -0.05367281245082792,
                                        0.1531558711397399, -0.001030634273454133, 0.0006584764419034815,
                                       -0.0016821925351926288, -0.004284529460797688, 0.030389771690123243,
                                       -0.040592118429752494, -0.1695472679986807, -0.20799422095574033, 0.045397975984119654,
                                       -0.03946672931050908, -0.16795588539580256, -0.20911090583076936, 0.0419854257806720])
mann_input.joint_velocities = np.array([0.0] * 26)



# mann_trajectory_generator_input = blf.ml.MANNAutoregressiveInput()
# mann_trajectory_generator_input.desired_future_base_trajectory = np.reshape(np.array([0.0, 0.0, 0.0, 0.12222222238779068,
#                                                                                      0.0, 0.2222222238779068, 0.0, 0.30000001192092896,
#                                                                                      0.0, 0.35555556416511536, 0.0, 0.3888888955116272,
#                                                                                      0.0, 0.4000000059604645]), (2, 7), order='F')
# mann_trajectory_generator_input.desired_future_base_velocities = np.reshape(np.array([0.0, 0.4000000059604645, 0.0, 0.4000000059604645, 0.0, 0.4000000059604645,
#                                                                                       0.0, 0.4000000059604645, 0.0, 0.4000000059604645, 0.0, 0.4000000059604645,0.0, 0.4000000059604645]),
#                                                                                       (2, 7), order='F')
# mann_trajectory_generator_input.desired_future_facing_directions = np.reshape(np.array([1.0, 0] * 7),
#                                                                                       (2, 7), order='F')

# print(mann_trajectory_generator_input.desired_future_base_trajectory)



initial_base_height = 0.7748

base_pose = manif.SE3.Identity()

quat = np.array([ 0, -0.0399893, 0, 0.9992001 ])
# quat = np.array([ 0, 0, 0,  1 ])
quat = quat / np.linalg.norm(quat)
base_pose = manif.SE3([0, 0, initial_base_height], quat)

left_foot = blf.contacts.EstimatedContact()
left_foot.is_active = True
left_foot.name = "left foot"
left_foot.index = ml.model().getFrameIndex("l_sole")


left_foot.switch_time = datetime.timedelta(seconds=0.0)
left_foot.pose = manif.SE3([0, 0.08, 0], manif.SO3.Identity().coeffs())

right_foot = blf.contacts.EstimatedContact()
right_foot.name = "right foot"
right_foot.index = ml.model().getFrameIndex("r_sole")
right_foot.is_active = True
right_foot.switch_time = datetime.timedelta(seconds=0.0)
right_foot.pose = manif.SE3([0, -0.08, 0], manif.SO3.Identity().coeffs())

mann_trajectory_generator.reset(mann_input, left_foot, right_foot, base_pose, manif.SE3Tangent.Zero())


input_builder = blf.ml.MANNAutoregressiveInputBuilder()
input_builder.initialize(params_joypad)

input_builder_input = blf.ml.MANNDirectionalInput()

# # Initialise YARP
yarp.Network.init()

# # Open and connect YARP port to retrieve joystick input
p_in = yarp.BufferedPortVector()
p_in.open("/joystick_in")
yarp.Network.connect("/joypad/goal:o", "/joystick_in")

input()

while(True):

    # Read from the input port
    res = p_in.read(shouldWait=True)

    input_builder_input.motion_direction = np.array([res[0], res[1]])
    input_builder_input.facing_direction = np.array([res[2], res[3]])

    input_builder.set_input(input_builder_input)
    input_builder.advance()

    mann_trajectory_generator.set_input(input_builder.get_output())

    assert mann_trajectory_generator.advance()

    mann_output = mann_trajectory_generator.get_output()
    viz.set_multibody_system_state(mann_output.base_pose.translation(),
                                    mann_output.base_pose.rotation(),
                                    mann_output.joint_positions)

#    blf.clock().sleep_for(datetime.timedelta(seconds=0.02))