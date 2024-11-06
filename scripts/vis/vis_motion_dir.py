"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Visualize motion library
"""

import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import joblib
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch
from poselib.poselib.skeleton.skeleton3d import SkeletonTree
from phc.utils.motion_lib_h1 import MotionLibH1
from phc.utils.motion_lib_stompy import MotionLibStompy
from phc.utils.flags import flags
from phc.utils.motion_lib_base import FixHeightMode
import time

flags.test = True
flags.im_eval = True


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


# simple asset descriptor for selecting from a list


class AssetDesc:

    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments


masterfoot = False
# masterfoot = True


asset_xml = "legged_gym/resources/robots/stompypro/robot_test.xml"
asset_urdf = "legged_gym/resources/robots/stompypro/robot_test.urdf"
asset_descriptors = [
    AssetDesc(asset_urdf, False),
]
sk_tree = SkeletonTree.from_mjcf(asset_xml)
asset_root = os.getcwd()
motion_file = os.path.join(asset_root, "data/stompy/amass_train.pkl")

# parse arguments
args = gymutil.parse_arguments(
    description="Joint monkey: Animate degree-of-freedom ranges",
    custom_parameters=[
        {
            "name": "--asset_id",
            "type": int,
            "default": 0,
            "help": "Asset id (0 - %d)" % (len(asset_descriptors) - 1),
        },
        {
            "name": "--speed_scale",
            "type": float,
            "default": 1.0,
            "help": "Animation speed scale",
        },
        {
            "name": "--show_axis",
            "action": "store_true",
            "help": "Visualize DOF axis",
        },
    ],
)

if args.asset_id < 0 or args.asset_id >= len(asset_descriptors):
    print(
        "*** Invalid asset_id specified.  Valid range is 0 to %d"
        % (len(asset_descriptors) - 1)
    )
    quit()

# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 30.0
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

if not args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(
    args.compute_device_id,
    args.graphics_device_id,
    args.physics_engine,
    sim_params,
)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load asset

asset_file = asset_descriptors[args.asset_id].file_name

asset_options = gymapi.AssetOptions()
asset_options.use_mesh_materials = True

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# set up the env grid
num_envs = 1
num_per_row = 5
spacing = 5
env_lower = gymapi.Vec3(-spacing, spacing, 0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(0, -10.0, 3)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cache useful handles
envs = []
actor_handles = []

num_dofs = gym.get_asset_dof_count(asset)
print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0, 0.0)
    pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    # set default DOF positions
    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

gym.prepare_sim(sim)


device = (
    torch.device("cuda", index=0)
    if torch.cuda.is_available()
    else torch.device("cpu")
)

motion_lib = MotionLibStompy(
    motion_file=motion_file,
    device=device,
    masterfoot_conifg=None,
    fix_height=False,
    multi_thread=False,
    mjcf_file=asset_xml,
)
num_motions = 10
curr_start = 0
motion_lib.load_motions(
    skeleton_trees=[sk_tree] * num_motions,
    gender_betas=[torch.zeros(18)] * num_motions,
    limb_weights=[np.zeros(10)] * num_motions,
    random_sample=False,
)
motion_keys = motion_lib.curr_motion_keys

num_motions = 3
curr_start = 0


current_dof = 0
speeds = np.zeros(num_dofs)

time_step = 0
rigidbody_state = gym.acquire_rigid_body_state_tensor(sim)
rigidbody_state = gymtorch.wrap_tensor(rigidbody_state)
rigidbody_state = rigidbody_state.reshape(num_envs, -1, 13)

actor_root_state = gym.acquire_actor_root_state_tensor(sim)
actor_root_state = gymtorch.wrap_tensor(actor_root_state)

gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT, "previous")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT, "next")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_G, "add")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_P, "print")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_T, "next_batch")
motion_id = 0
motion_acc = set()
env_ids = torch.arange(num_envs).int().to(args.sim_device)
dict_new = joblib.load("test.pkl")
while not gym.query_viewer_has_closed(viewer):
    motion_len = motion_lib.get_motion_length(motion_id).item()
    motion_time = time_step % motion_len
    # step the physics

    if args.show_axis:
        gym.clear_lines(viewer)

    root_pos = torch.zeros((1, 3)).cuda()
    root_pos[:, :] = dict_new["root_pos"][
        int(time_step / dt) % dict_new["root_pos"].shape[0]
    ]
    root_pos[:, 2] += 0.2
    root_rot = torch.zeros((1, 4)).cuda()
    root_rot[:, :] = dict_new["root_rot"][
        int(time_step / dt) % dict_new["root_rot"].shape[0]
    ]
    root_vel = torch.zeros((1, 3)).cuda()
    root_ang_vel = torch.zeros((1, 3)).cuda()

    root_states = torch.cat(
        [root_pos, root_rot, root_vel, root_ang_vel], dim=-1
    ).repeat(num_envs, 1)
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))
    gym.set_actor_root_state_tensor_indexed(
        sim,
        gymtorch.unwrap_tensor(root_states),
        gymtorch.unwrap_tensor(env_ids),
        len(env_ids),
    )

    gym.refresh_actor_root_state_tensor(sim)

    dof_pos = torch.zeros((1, 19)).cuda()
    dof_pos[0, :] = dict_new["dof"][
        int(time_step / dt) % dict_new["dof"].shape[0]
    ]

    dof_state = (
        torch.stack([dof_pos, torch.zeros_like(dof_pos)], dim=-1)
        .squeeze()
        .repeat(num_envs, 1)
    )
    gym.set_dof_state_tensor_indexed(
        sim,
        gymtorch.unwrap_tensor(dof_state),
        gymtorch.unwrap_tensor(env_ids),
        len(env_ids),
    )

    gym.simulate(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)
    time_step += dt
    time.sleep(dt)

    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "previous" and evt.value > 0:
            motion_id = (motion_id - 1) % num_motions
            print(
                f"Motion ID: {motion_id}. Motion length: {motion_len:.3f}. Motion Name: {motion_keys[motion_id]}"
            )
        elif evt.action == "next" and evt.value > 0:
            motion_id = (motion_id + 1) % num_motions
            print(
                f"Motion ID: {motion_id}. Motion length: {motion_len:.3f}. Motion Name: {motion_keys[motion_id]}"
            )
        elif evt.action == "add" and evt.value > 0:
            motion_acc.add(motion_keys[motion_id])
            print(f"Adding motion {motion_keys[motion_id]}")
        elif evt.action == "print" and evt.value > 0:
            print(motion_acc)
        elif evt.action == "next_batch" and evt.value > 0:
            curr_start += num_motions
            motion_lib.load_motions(
                skeleton_trees=[sk_tree] * num_motions,
                gender_betas=[torch.zeros(17)] * num_motions,
                limb_weights=[np.zeros(10)] * num_motions,
                random_sample=False,
                start_idx=curr_start,
            )
            motion_keys = motion_lib.curr_motion_keys
            print(f"Next batch {curr_start}")

        time_step = 0
print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
