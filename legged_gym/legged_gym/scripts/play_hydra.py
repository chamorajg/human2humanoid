import os
import sys
from datetime import datetime

import cv2
import hydra
import isaacgym
import numpy as np
import torch
from easydict import EasyDict
from isaacgym import gymapi
from omegaconf import DictConfig, OmegaConf
from termcolor import colored
from tqdm import tqdm

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import (
    Logger,
    export_policy_as_jit,
    export_policy_as_onnx,
    get_args,
    task_registry,
)
from legged_gym.utils.helpers import class_to_dict

NOROSPY = False
try:
    import rospy
except:
    NOROSPY = True
# from std_msgs.msg import String, Header, Float64MultiArray

command_state = {
    "vel_forward": 0.0,
    "vel_side": 0.0,
    "orientation": 0.0,
}

override = False

EXPORT_ONNX = False


def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    shared_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o: (d1[o], d2[o]) for o in shared_keys if d1[o] != d2[o]}
    same = set(o for o in shared_keys if d1[o] == d2[o])
    return added, removed, modified, same


class VideoRecorder:
    """Utility class for logging evaluation videos."""

    def __init__(self, root_dir, render_size=384, fps=25):
        self.save_dir = (
            os.path.join(root_dir, "eval_video") if root_dir else None
        )
        self.render_size = render_size
        self.fps = fps
        self.frames_length = 0
        self.enabled = False
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Creates a directory to store videos.
        os.makedirs(self.save_dir, exist_ok=True)
        self.dir = os.path.join(self.save_dir, now + ".mp4")
        self.video = cv2.VideoWriter(self.dir, fourcc, float(self.fps), (1920, 1080))

    def init(self, env, stompypro, enabled=True):
        self.frames_length = 0
        self.enabled = self.save_dir and enabled
        self.record(env, stompypro)

    def record(self, env, stompypro):
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        img = env.gym.get_camera_image(
            env.sim, env.envs[0], stompypro, gymapi.IMAGE_COLOR
        )
        img = np.reshape(img, (1080, 1920, 4))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(os.path.join(self.save_dir, f"image_{self.frames_length}.png"), img)
        self.video.write(img)
        self.frames_length += 1

        if self.frames_length == 25:
            self.video.release()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.dir = os.path.join(self.save_dir, now + ".mp4")
            self.video = cv2.VideoWriter(
                self.dir, fourcc, float(self.fps), (1920, 1080)
            )
            self.frames_length = 0

    def save(
        self,
    ):
        self.video.release()


@hydra.main(
    version_base=None,
    config_path="../cfg",
    config_name="config_base",
)
def play(cfg_hydra: DictConfig) -> None:
    cfg_hydra = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
    cfg_hydra.physics_engine = gymapi.SIM_PHYSX

    env_cfg, train_cfg = cfg_hydra, cfg_hydra.train

    # if not env_cfg.train_velocity_estimation:
    env_cfg.env.num_envs = 1
    env_cfg.viewer.debug_viz = False
    env_cfg.motion.visualize = False
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.mesh_type = "trimesh"
    if env_cfg.terrain.mesh_type == "trimesh":
        env_cfg.terrain.terrain_types = [
            "flat",
            "rough",
            "low_obst",
        ]  # do not duplicate!
        env_cfg.terrain.terrain_proportions = [1.0, 0.0, 0.0]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.env.episode_length_s = 20
    env_cfg.domain_rand.randomize_rfi_lim = False
    env_cfg.domain_rand.randomize_pd_gain = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_ctrl_delay = False
    env_cfg.domain_rand.ctrl_delay_step_range = [1, 3]

    env_cfg.env.test = True

    if env_cfg.motion.realtime_vr_keypoints:
        env_cfg.asset.terminate_by_1time_motion = False
        env_cfg.asset.terminate_by_ref_motion_distance = False
        rospy.init_node("avppose_subscriber")
        from avp_pose_subscriber import AVPPoseInfo

        avpposeinfo = AVPPoseInfo()
        rospy.Subscriber(
            "avp_pose",
            Float64MultiArray,
            avpposeinfo.avp_callback,
            queue_size=1,
        )
    if cfg_hydra.joystick:
        env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
        env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
        env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
        from pynput import keyboard

        from legged_gym.utils import key_response_fn

    # prepare environment
    env, _ = task_registry.make_env_hydra(
        name=cfg_hydra.task, hydra_cfg=cfg_hydra, env_cfg=env_cfg
    )

    camera_properties = gymapi.CameraProperties()
    camera_properties.width = 1920
    camera_properties.height = 1080
    stompypro = env.gym.create_camera_sensor(env.envs[0], camera_properties)

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 4  # which joint is used for logging
    stop_state_log = 200  # number of steps before plotting states
    stop_rew_log = (
        env.max_episode_length + 1
    )  # number of steps before print average episode rewards

    obs = env.get_observations()

    video = VideoRecorder(
        "/home/chandramouli/human2humanoid/legged_gym/output/stompypro"
    )

    if env_cfg.motion.realtime_vr_keypoints:
        init_root_pos = env._rigid_body_pos[..., 0, :].clone()
        init_avp_pos = avpposeinfo.avp_pose.copy()
        init_root_offset = init_root_pos[0, :2] - init_avp_pos[2, :2]
    # load policy
    train_cfg.runner.resume = True

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=cfg_hydra.task, args=cfg_hydra, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    exported_policy_name = str(
        task_registry.loaded_policy_path.split("/")[-2]
    ) + str(task_registry.loaded_policy_path.split("/")[-1])
    print("Loaded policy from: ", task_registry.loaded_policy_path)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(
            ppo_runner.alg.actor_critic, path, exported_policy_name
        )
        print(
            "Exported policy as jit script to: ",
            os.path.join(path, exported_policy_name),
        )
    if EXPORT_ONNX:
        exported_onnx_name = exported_policy_name.replace(".pt", ".onnx")
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_onnx(
            ppo_runner.alg.actor_critic,
            path,
            exported_onnx_name,
            onnx_num_observations=env_cfg.env.num_observations,
        )
        print(
            "Exported policy as onnx to: ",
            os.path.join(path, exported_onnx_name),
        )

    if cfg_hydra.joystick:
        print(colored("joystick on", "green"))
        key_response = key_response_fn(mode="vel")

        def on_press(key):
            global command_state
            try:
                # print(key.char)
                key_response(key, command_state, env)
            except AttributeError:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    i = 0

    if env_cfg.get("record_video", False):
        video.init(env, stompypro, enabled=True)
    pbar = tqdm(total=env.max_episode_length + 1)
    while (not NOROSPY and not rospy.is_shutdown()) or (NOROSPY):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        if env_cfg.get("record_video", False):
            video.record(env, stompypro)

        if env_cfg.motion.realtime_vr_keypoints:
            avpposeinfo.check()
            keypoints_pos = avpposeinfo.avp_pose.copy()
            keypoints_pos[:, 0] += init_root_offset[0].item()
            keypoints_pos[:, 1] += init_root_offset[1].item()
            keypoints_vel = avpposeinfo.avp_vel.copy()
            env._update_realtime_vr_keypoints(keypoints_pos, keypoints_vel)
        if override:
            obs[:, 9] = 0.5
            obs[:, 10] = 0.0
            obs[:, 11] = 0.0

        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": actions[robot_index, joint_index].item()
                    * env.cfg.control.action_scale
                    + env.default_dof_pos[robot_index, joint_index].item(),
                    "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "contact_forces_z": env.contact_forces[
                        robot_index, env.feet_indices, 2
                    ]
                    .cpu()
                    .numpy(),
                }
            )
        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            pass
        i += 1
        pbar.update(1)

    if env_cfg.get("record_video", False):
        video.save()


if __name__ == "__main__":
    EXPORT_POLICY = False
    RECORD_FRAMES = True
    MOVE_CAMERA = False
    play()
