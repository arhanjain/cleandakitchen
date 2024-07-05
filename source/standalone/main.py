
import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="test")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

########################################

import torch
import numpy as np
import gymnasium as gym
import omni.isaac.lab.sim as sim_utils

import omni.isaac.lab_tasks
from omni.isaac.lab_tasks.utils import parse_env_cfg

from omni.isaac.lab.assets.asset_base_cfg import AssetBaseCfg
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG
from omni.isaac.lab.sensors.camera import Camera, CameraCfg
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth
from m2t2_utils import save_data
import sys
sys.path.append('/home/jacob/projects/cleandakitchen/M2T2')
from demo import load_and_predict
from planner import MotionPlanner

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

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
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 1.05))
    )

    #cube 1
    cube_one = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube1",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg()
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.3, 1.2))
    )

    # articulation
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.0, 0.0, 1.05)

    # camera
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraSensor",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    camera = scene["camera"]
    # Controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    
    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm
    ee_goals = [
        [0.5, 0, 0.7, 0.707, 0, 0.707, 0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    # Specify robot-specific parameters
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]


    sim_dt = sim.get_physics_dt()
    # Define simulation stepping
    count = 0
    # Saving pointcloud info
    frame = 0
    output_dir = 'output_data'
    # Simulation loop
    planner = MotionPlanner(scene, sim.device)
    goal = None
    plan = None
    #reset
    count = 0
    # reset joint state
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    # reset actions
    ik_commands[:] = ee_goals[current_goal_idx]
    joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
    # reset cont     
    diff_ik_controller.reset()
    diff_ik_controller.set_command(ik_commands)

    while simulation_app.is_running():
        camera.update(dt = sim_dt)
        # doesn't actually take in pointcloud, instead it needs the corresponding files.
        save_data(camera, output_dir, frame)
        import hydra
        from omegaconf import OmegaConf
        cfg = OmegaConf.load("/home/jacob/projects/cleandakitchen/M2T2/config.yaml")
        data_dir = output_dir
        data, outputs = load_and_predict(data_dir, cfg) # data is used to determine pick and place, visualization, etc

        frame += 1
        if outputs and 'grasps' in outputs:
            grasping_point = outputs['grasps'][0][0]
            grasping_orientation = outputs['grasps'][0][3:]

            goal = torch.tensor([grasping_point[0], grasping_point[1], grasping_point[2], grasping_orientation[0], grasping_orientation[1], grasping_orientation[2], grasping_orientation[3]], device=sim.device)

            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            joint_vel = robot.data.joint_vel[:, robot_entity_cfg.joint_ids]
            joint_names = robot.data.joint_names

            plan = planner.plan(joint_pos, joint_vel, joint_names, goal.squeeze())
            for joint_pos in plan.position:
                for j in range(20):
                    robot.set_joint_position_target(joint_pos, joint_ids=robot_entity_cfg.joint_ids)
                    scene.write_data_to_sim()
                    sim.step()
                    count += 1
                    scene.update(sim_dt)
                    ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
                    ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
                    goal_marker.visualize(goal[:, 0:3], goal[:, 3:7])

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = TableTopSceneCfg(num_envs=1, env_spacing=2.0)
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
