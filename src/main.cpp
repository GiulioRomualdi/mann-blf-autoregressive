#include <BipedalLocomotion/ML/MANNTrajectoryGenerator.h>
#include <BipedalLocomotion/ParametersHandler/TomlImplementation.h>
#include <BipedalLocomotion/Contacts/Contact.h>
#include <BipedalLocomotion/System/Clock.h>
#include <BipedalLocomotion/ReducedModelControllers/CentroidalMPC.h>
#include <BipedalLocomotion/ContinuousDynamicalSystem/CentroidalDynamics.h>
#include <BipedalLocomotion/ContinuousDynamicalSystem/ForwardEuler.h>

#include <manif/SE3.h>

#include <iDynTree/ModelIO/ModelLoader.h>
#include <iDynTree/Model/Model.h>

#include <string>
#include <memory>
#include <chrono>

using namespace BipedalLocomotion;
using namespace std::chrono_literals;

void updateContactPhaseList(
    const std::map<std::string, BipedalLocomotion::Contacts::PlannedContact>& nextPlannedContacts,
    BipedalLocomotion::Contacts::ContactPhaseList& phaseList)
{
    auto newList = phaseList.lists();
    for (const auto& [key, contact] : nextPlannedContacts)
    {
        auto it = newList.at(key).getPresentContact(contact.activationTime);
        if (it != newList.at(key).cend())
        {
            newList.at(key).editContact(it, contact);
        }
    }

    phaseList.setLists(newList);
}

int main()
{

    std::ofstream myFile;
    myFile.open("data.txt");

    ML::MANNTrajectoryGenerator generator;
    const std::string robot_model_path = "/home/gromualdi/robot-install/share/ergoCub/robots/ergoCubGazeboV1/model.urdf";
    const std::string param_file_mann = "/home/gromualdi/robot-code/test-mann-blf/config/config_mann.toml";
    const std::string param_file_mpc = "/home/gromualdi/robot-code/test-mann-blf/config/config_mpc.toml";

    auto params = std::make_shared<ParametersHandler::TomlImplementation>();
    params->setFromFile(param_file_mann);

    std::vector<std::string> jointsList;
    iDynTree::ModelLoader ml;
    params->getParameter("joints_list", jointsList);
    ml.loadReducedModelFromFile(robot_model_path, jointsList);
    generator.setRobotModel(ml.model());
    generator.initialize(params);

    Eigen::VectorXd jointPositions(26);
    jointPositions << -0.10922017141063572, 0.05081325960010118, 0.06581966291990003, -0.0898053099824925, -0.09324922528169599, -0.05110058859172172,
        -0.11021232812838086, 0.054291515925228385, 0.0735575862560208, -0.09509332143185895, -0.09833823347493076, -0.05367281245082792,
        0.1531558711397399, -0.001030634273454133, 0.0006584764419034815,
        -0.0016821925351926288, -0.004284529460797688, 0.030389771690123243,
        -0.040592118429752494, -0.1695472679986807, -0.20799422095574033, 0.045397975984119654,
        -0.03946672931050908, -0.16795588539580256, -0.20911090583076936, 0.0419854257806720;

    ML::MANNTrajectoryGeneratorInput generatorInput;
    generatorInput.desiredFutureBaseTrajectory.resize(2, 7);
    generatorInput.desiredFutureBaseVelocities.resize(2, 7);
    generatorInput.desiredFutureFacingDirections.resize(2, 7);
    generatorInput.desiredFutureBaseTrajectory << 0, 0.12, 0.22, 0.3, 0.35, 0.39, 0.4,
        0, 0, 0, 0, 0, 0, 0;

    for (int i = 0; i < generatorInput.desiredFutureFacingDirections.cols(); i++)
    {
        generatorInput.desiredFutureFacingDirections.col(i) << 1.0, 0;
        generatorInput.desiredFutureBaseVelocities.col(i) << 0.4, 0;
    }
    generatorInput.mergePointIndex = 0;

    std::cerr << "_-----------" << std::endl;
    std::cerr << generatorInput.desiredFutureBaseTrajectory << std::endl;

    std::cerr << "_-----------" << std::endl;
    std::cerr << generatorInput.desiredFutureFacingDirections << std::endl;

    std::cerr << "_-----------" << std::endl;
    std::cerr << generatorInput.desiredFutureBaseVelocities << std::endl;
    std::cerr << "_-----------" << std::endl;

    manif::SE3d basePose = manif::SE3d(Eigen::Vector3d{0, 0, 0.7748},
                                       Eigen::AngleAxis(0.0, Eigen::Vector3d::UnitY()));

    Contacts::EstimatedContact leftFoot, rightFoot;
    leftFoot.isActive = true;
    leftFoot.name = "left_foot";
    leftFoot.index = ml.model().getFrameIndex("l_sole");
    leftFoot.switchTime = 0s;
    leftFoot.pose = manif::SE3d(Eigen::Vector3d{0, 0.08, 0}, manif::SO3d::Identity());

    rightFoot.isActive = true;
    rightFoot.name = "right_foot";
    rightFoot.index = ml.model().getFrameIndex("r_sole");
    rightFoot.switchTime = 0s;
    rightFoot.pose = manif::SE3d(Eigen::Vector3d{0, -0.08, 0}, manif::SO3d::Identity());

    generator.setInitialState(jointPositions, leftFoot, rightFoot, basePose, 0s);

    auto params_mpc = std::make_shared<ParametersHandler::TomlImplementation>();
    params_mpc->setFromFile(param_file_mpc);

    ReducedModelControllers::CentroidalMPC mpc;
    mpc.initialize(params_mpc);

    auto system = std::make_shared<ContinuousDynamicalSystem::CentroidalDynamics>();
    constexpr std::chrono::nanoseconds integratorStepTime = 40ms;
    ContinuousDynamicalSystem::ForwardEuler<ContinuousDynamicalSystem::CentroidalDynamics> integrator;
    integrator.setIntegrationStep(integratorStepTime);
    integrator.setDynamicalSystem(system);

    std::chrono::nanoseconds time = 0s;

    int numberOfSupport = 2;
    std::map<std::string, Contacts::PlannedContact> nextPlannedContact;
    Eigen::Vector3d comPosition, comVelocity, angularMomentum;
    for (int i = 0; i < 20000; i++)
    {
        auto begin = BipedalLocomotion::clock().now();
        generator.setInput(generatorInput);
        std::cerr << "---> new iteration of mann" << std::endl;
        generator.advance();

        if (i == 0)
        {
            system->setState({generator.getOutput().comTrajectory.col(0), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()});
        }

        comPosition = system->getState().get<0>();
        comVelocity = system->getState().get<1>();
        angularMomentum = system->getState().get<2>();
        generatorInput.mergePointIndex = 1;
        mpc.setState(comPosition, comVelocity, angularMomentum);
        mpc.setReferenceTrajectory(generator.getOutput().comTrajectory, generator.getOutput().angularMomentumTrajectory / 56.0);

        auto phaseList = generator.getOutput().phaseList;

        if(numberOfSupport == 1 && phaseList.getPresentPhase(time)->activeContacts.size() == 2)
        {
            numberOfSupport = 2;
            nextPlannedContact = mpc.getOutput().nextPlannedContact;
        }
        else if (numberOfSupport == 2 && phaseList.getPresentPhase(time)->activeContacts.size() == 1)
        {
            numberOfSupport = 1;
        }

        updateContactPhaseList(nextPlannedContact, phaseList);

        mpc.setContactPhaseList(phaseList);
        mpc.advance();
        auto end = BipedalLocomotion::clock().now();



        auto elapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
/*         BipedalLocomotion::log()->warn("time to compute trajectory {}", std::chrono::duration<double>(elapsedTime).count());

        for(const auto& [key, list]: generator.getOutput().phaseList.lists())
        {
            log()->info("--------------> {}", key);
            for( const BipedalLocomotion::Contacts::PlannedContact& contact : list)
            {
                log()->info("activation{}s  deactivation {}s",
                 std::chrono::duration<double>(contact.activationTime).count(),
                std::chrono::duration<double>(contact.deactivationTime).count());
            }
        }
 */

        if (i == 0)
        {
            for (const auto &[key, contact] : mpc.getOutput().contacts)
            {
                myFile << key << "_pos_x " << key << "_pos_y " << key << "_pos_z ";
                for (int j = 0; j < contact.corners.size(); j++)
                {
                    myFile << key << "_" << j << "_x"
                           << " " << key << "_" << j << "_y"
                           << " " << key << "_" << j << "_z ";
                }

                myFile << key << "_next_pos_x " << key << "_next_pos_y " << key << "_next_pos_z ";
            }
            myFile << "com_x com_y com_z des_com_x des_com_y des_com_z ang_x ang_y ang_z "
                      "elapsed_time"
                   << std::endl;
        }

        for (const auto &[key, contact] : mpc.getOutput().contacts)
        {
            myFile << contact.pose.translation().transpose() << " ";
            for (const auto &corner : contact.corners)
            {
                myFile << corner.force.transpose() << " ";
            }

            auto nextPlannedContact = mpc.getOutput().nextPlannedContact.find(key);
            if (nextPlannedContact == mpc.getOutput().nextPlannedContact.end())
            {
                myFile << 0.0 << " " << 0.0 << " " << 0.0 << " ";
            }
            else
            {
                myFile << nextPlannedContact->second.pose.translation().transpose() << " ";
            }
        }
        myFile << comPosition.transpose() << " " << generator.getOutput().comTrajectory.col(0).transpose() << " "
               << angularMomentum.transpose() << " " << elapsedTime.count() << std::endl;

        Eigen::Vector3d externalWrench = Eigen::Vector3d::Zero();
        if (46 <= i && i <= 50)
        {
            std::cerr << "external wrench" << std::endl;
            externalWrench(1) = -5;
        }
        system->setControlInput({mpc.getOutput().contacts, externalWrench});
        integrator.integrate(0s, integratorStepTime);

        time += integratorStepTime;
    }

    return 0;
}
