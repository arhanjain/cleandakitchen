import argparse
from omni.isaac.lab.app import AppLauncher


parser = argparse.ArgumentParser(description="Testing cuRobo motion planner for Panda.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import argparse
from omni.isaac.lab.app import AppLauncher
import torch
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG

# For now generate random joint positions
def generate_random_joint_positions(robot, joint_limits):
    joint_positions = torch.zeros(robot.data.joint_pos.shape, device=robot.device)
    for i in range(len(joint_limits)):
        joint_positions[:, i] = torch.FloatTensor(robot.data.num_envs).uniform_(
            joint_limits[i][0], joint_limits[i][1]
        ).to(robot.device)
    return joint_positions

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a tabletop scene."""
    
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]

    # Get joint limits
    joint_limits = robot.get_joint_limits()

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 150 == 0:
            # reset time
            count = 0
            # Generate random joint positions
            random_joint_positions = generate_random_joint_positions(robot, joint_limits)
            # reset joint state
            robot.write_joint_state_to_sim(random_joint_positions, robot.data.default_joint_vel)
            robot.reset()
        else:
            # apply random actions
            random_joint_positions = generate_random_joint_positions(robot, joint_limits)
            robot.set_joint_position_target(random_joint_positions)

        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
