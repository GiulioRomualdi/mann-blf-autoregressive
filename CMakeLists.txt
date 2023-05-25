cmake_minimum_required(VERSION 3.16.0)

project(myproject)
find_package(BipedalLocomotionFramework REQUIRED)

add_executable(example src/main.cpp)
target_link_libraries(example  BipedalLocomotion::ML BipedalLocomotion::ReducedModelControllers
BipedalLocomotion::ParametersHandlerTomlImplementation
BipedalLocomotion::ContinuousDynamicalSystem)