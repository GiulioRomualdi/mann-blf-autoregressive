
from idyntree.visualize import MeshcatVisualizer
import idyntree.bindings as idyn
from pathlib import Path
import numpy as np
import manifpy as manif

import bipedal_locomotion_framework as blf
import datetime
import time

robot_model_path = "/home/gromualdi/robot-install/share/ergoCub/robots/ergoCubGazeboV1/model.urdf"
config_path = Path(__file__).parent / "config" / "config_mann.toml"

params = blf.parameters_handler.TomlParametersHandler()
params.set_from_file(str(config_path))

ml = idyn.ModelLoader()
ml.loadReducedModelFromFile(robot_model_path,params.get_parameter_vector_string("joints_list"))

viz = MeshcatVisualizer()
viz.load_model(ml.model())

mann_trajectory_generator = blf.ml.MANNTrajectoryGenerator()
mann_trajectory_generator.set_robot_model(ml.model())
mann_trajectory_generator.initialize(params)


# reset the mann with the initial valuereshape(a, newshape, order='C')[source]


joint_positions = np.array([-0.10922017141063572, 0.05081325960010118, 0.06581966291990003, -0.0898053099824925, -0.09324922528169599, -0.05110058859172172,
                                       -0.11021232812838086, 0.054291515925228385,0.0735575862560208, -0.09509332143185895, -0.09833823347493076, -0.05367281245082792,
                                        0.1531558711397399, -0.001030634273454133, 0.0006584764419034815,
                                       -0.0016821925351926288, -0.004284529460797688, 0.030389771690123243,
                                       -0.040592118429752494, -0.1695472679986807, -0.20799422095574033, 0.045397975984119654,
                                       -0.03946672931050908, -0.16795588539580256, -0.20911090583076936, 0.0419854257806720])

mann_trajectory_generator_input = blf.ml.MANNTrajectoryGeneratorInput()
# mann_trajectory_generator_input.desired_future_base_trajectory = np.reshape(np.array([0.0, 0.0, 0.0, 0.12222222238779068,
#                                                                                      0.0, 0.2222222238779068, 0.0, 0.30000001192092896,
#                                                                                      0.0, 0.35555556416511536, 0.0, 0.3888888955116272,
#                                                                                      0.0, 0.4000000059604645]), (2, 7), order='F')
# mann_trajectory_generator_input.desired_future_base_velocities = np.reshape(np.array([0.0, 0.4000000059604645, 0.0, 0.4000000059604645, 0.0, 0.4000000059604645,
#                                                                                       0.0, 0.4000000059604645, 0.0, 0.4000000059604645, 0.0, 0.4000000059604645,0.0, 0.4000000059604645]),
#                                                                                       # (2, 7), order='F')
mann_trajectory_generator_input.desired_future_facing_directions = np.reshape(np.array([1.0, 0] * 7),
                                                                                      (2, 7), order='F')


mann_trajectory_generator_input.desired_future_base_trajectory = np.reshape(np.array([0.0, 0.0, 0.22, 0.0,
                                                                                     0.32, 0.0, 0.33, 0.0,
                                                                                     0.35, 0.0, 0.39, 0.0,
                                                                                     0.4, 0.0]), (2, 7), order='F')
mann_trajectory_generator_input.desired_future_base_velocities = np.reshape(np.array([0.40, 0.0] * 7),
                                                                                      (2, 7), order='F')

# mann_trajectory_generator_input.desired_future_base_trajectory = np.reshape(np.array([0.0, 0.0] * 7), (2, 7), order='F')
# mann_trajectory_generator_input.desired_future_base_velocities = np.reshape(np.array([0.0, 0.0] * 7),
#                                                                                       (2, 7), order='F')
# mann_trajectory_generator_input.desired_future_facing_directions = np.reshape(np.array([1.0, 0] * 7),
#                                                                                       (2, 7), order='F')
mann_trajectory_generator_input.merge_point_index = 0


initial_base_height = 0.7748

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

mann_trajectory_generator.set_initial_state(joint_positions, left_foot, right_foot, base_pose, datetime.timedelta(0))

input()

for j in range(1):
  mann_trajectory_generator.set_input(mann_trajectory_generator_input)
  mann_trajectory_generator.advance()
  output = mann_trajectory_generator.get_output()
  mann_trajectory_generator_input.merge_point_index = 1

  for i in range(len(output.joint_positions)):
    input()
    viz.set_multibody_system_state(output.base_pose[i].translation(),
                                  output.base_pose[i].rotation(),
                                  output.joint_positions[i])
    mann_trajectory_generator_input.merge_point_index = 1

  # time.sleep(0.5)

# mann_trajectory_generator_input.desired_future_base_trajectory = np.reshape(np.array([0.0, 0.0, 0.0, 0.12222222238779068,
#                                                                                      0.0, 0.2222222238779068, 0.0, 0.30000001192092896,
#                                                                                      0.0, 0.35555556416511536, 0.0, 0.3888888955116272,
#                                                                                      0.0, 0.4000000059604645]), (2, 7), order='F')
# mann_trajectory_generator_input.desired_future_base_velocities = np.reshape(np.array([0.0, 0.4000000059604645, 0.0, 0.4000000059604645, 0.0, 0.4000000059604645,
#                                                                                       0.0, 0.4000000059604645, 0.0, 0.4000000059604645, 0.0, 0.4000000059604645,0.0, 0.4000000059604645]),
#                                                                                       (2, 7), order='F')
# mann_trajectory_generator_input.desired_future_facing_directions = np.reshape(np.array([6.123234262925839e-17, 1.0, 6.123234262925839e-17, 1.0, 6.123234262925839e-17, 1.0,
#                                                                                         6.123234262925839e-17, 1.0, 6.123234262925839e-17, 1.0, 6.123234262925839e-17, 1.0,
#                                                                                         6.123234262925839e-17, 1.0]),
#                                                                                       (2, 7), order='F')


# mann_trajectory_generator_input.desired_future_base_trajectory = np.reshape(np.array([0.0, 0.0] * 7), (2, 7), order='F')
# mann_trajectory_generator_input.desired_future_base_velocities = np.reshape(np.array([0.0, 0.0] * 7),
#                                                                                       (2, 7), order='F')
# mann_trajectory_generator_input.desired_future_facing_directions = np.reshape(np.array([1.0, 0] * 7),
#                                                                                       (2, 7), order='F')
# start = time.time()
# mann_trajectory_generator.set_input(mann_trajectory_generator_input)
# mann_trajectory_generator.advance()
# end = time.time()

# output = mann_trajectory_generator.get_output()

# for i in range(len(output.joint_positions)):
#   viz.set_multibody_system_state(output.base_pose[i].translation(),
#                                  output.base_pose[i].rotation(),
#                                  output.joint_positions[i])
#   if i == 0:
#     input()
