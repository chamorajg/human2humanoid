# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from resources.robots.stompypro.joints import Robot

class StompyProTeleopCfg( LeggedRobotCfg ):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, Robot.height]
        rot = Robot.rotation
        default_joint_angles = {k: 0.0 for k in Robot.all_joints()}

        default_positions = Robot.default_standing()
        for joint in default_positions:
            default_joint_angles[joint] = default_positions[joint]

    class domain_rand ( LeggedRobotCfg.domain_rand ):
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 0.5
        
        randomize_friction = True
        # randomize_friction = False
        friction_range = [-0.6, 1.2]
        
        randomize_base_mass = False # replaced by randomize_link_mass
        added_mass_range = [-5., 10.]


        randomize_base_com = True
        class base_com_range: #kg
            x = [-0.1, 0.1]
            y = [-0.1, 0.1]
            z = [-0.1, 0.1]

        randomize_link_mass = True
        link_mass_range = [0.7, 1.3] # *factor
        randomize_link_body_names = [
            'L_buttock', 'L_leg', 'L_thigh', 
            'R_buttock', 'R_leg', 'R_leg',  'torso_link',]

        randomize_pd_gain = True
        kp_range = [0.75, 1.25]
        kd_range = [0.75, 1.25]


        randomize_torque_rfi = True
        rfi_lim = 0.1
        randomize_rfi_lim = True
        rfi_lim_range = [0.5, 1.5]

        randomize_motion_ref_xyz = True
        motion_ref_xyz_range = [[-0.02, 0.02],[-0.02, 0.02],[-0.05, 0.05]]
        
        randomize_ctrl_delay = True
        ctrl_delay_step_range = [1, 3] # integer max real delay is 90ms

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 80.}  # [N*m/rad]
        damping = {'joint': 1}     # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class noise ( LeggedRobotCfg.noise ):
        add_noise = True # False for teleop sim right now
        noise_level = 1.0 # scales other values
        class noise_scales:
            base_z = 0.05
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.2
            lin_acc = 0.2 # ???????????????
            ang_vel = 0.5
            gravity = 0.1
            in_contact = 0.1
            height_measurements = 0.05
            body_pos = 0.01 # body pos in cartesian space: 19x3
            body_lin_vel = 0.01 # body velocity in cartesian space: 19x3
            body_rot = 0.001 # 6D body rotation 
            delta_base_pos = 0.05
            delta_heading = 0.1
            last_action = 0.0
            
            ref_body_pos = 0.10
            ref_body_rot = 0.01
            ref_body_vel = 0.01
            ref_lin_vel = 0.01
            ref_ang_vel = 0.01
            ref_dof_pos = 0.01
            ref_dof_vel = 0.01
            ref_gravity = 0.01
    class sim ( LeggedRobotCfg.sim ):
        dt = 0.005  #   1/60.
    class terrain ( LeggedRobotCfg.terrain ):
        # mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        # mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        # border_size = 25 # [m] # for play only
        curriculum = False
        # curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        # measure_heights = False # keep it False
        # measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        # measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        measure_heights = False 
        measured_points_x = [ 0.] # 1mx1.6m rectangle (without center line)
        measured_points_y = [ 0.]

        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 9 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        terrain_types = ['flat', 'rough', 'low_obst', 'smooth_slope', 'rough_slope']  # do not duplicate!
        terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
        # terrain_proportions = [0.2, 0.6, 0.2, 0., 0.]
        # terrain_proportions = [1,, 0., 0., 0., 0.]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/stompypro/robot_test.xml'
        torso_name = "torso"
        foot_name = ["foot"]
        penalize_contacts_on = []
        terminate_after_contacts_on = ["torso", "clav", "calf", "thigh" ]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class commands ( LeggedRobotCfg.commands ):
        curriculum = False
        max_curriculum = 0.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [.0, .0] # min max [m/s]
            lin_vel_y = [.0, .0]   # min max [m/s]
            ang_vel_yaw = [.0, .0]    # min max [rad/s]
            heading = [.0, .0]

    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            torques = -9e-5*1.25
            torque_limits = -2e-1*1.25
            dof_acc = -8.4e-6*1.25 #-8.4e-6   -4.2e-7 #-3.5e-8
            dof_vel = -0.003*1.25 # -0.003
            # action_rate = -0.6 # -0.6  # -0.3 # -0.3 -0.12 -0.01
            lower_action_rate = -0.9*1.25 # -0.6  # -0.3 # -0.3 -0.12 -0.01
            upper_action_rate = -0.05*1.25 # -0.6  # -0.3 # -0.3 -0.12 -0.01
            dof_pos_limits = -100.0*1.25
            termination = -200*1.25
            feet_contact_forces = -0.10*1.25
            stumble = -1000.0*1.25
            feet_air_time_teleop = 800.0*1.25
            slippage = -30.0*1.25
            feet_ori = -50.0*1.25
            orientation = -0.0
            teleop_selected_joint_position = 32 # 5.0
            teleop_selected_joint_vel = 16 # 5.
            
            teleop_body_position = 0.0 # 6 keypoint
            teleop_body_position_extend = 40.0 # 8 keypoint
            teleop_body_position_extend_small_sigma = 0.0 # 8 keypoint
            teleop_body_rotation = 20.0
            teleop_body_vel = 8.0
            teleop_body_ang_vel = 8.0
            
            # slippage = -1.

        
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.85 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.85
        soft_torque_limit = 0.85

        max_contact_force = 500.

        teleop_joint_pos_sigma = 0.5
        teleop_joint_vel_sigma = 10.
        teleop_body_pos_sigma = 0.5 # 0.01
        teleop_body_pos_small_sigma = 0.01
        teleop_body_pos_lower_weight = 0.5
        teleop_body_pos_upper_weight = 1.0
        teleop_body_rot_sigma = 0.1
        teleop_body_vel_sigma = 10.
        teleop_body_ang_vel_sigma = 10.
        teleop_body_rot_selection = ['torso_link']
        teleop_body_vel_selection = ['torso_link']
        teleop_body_pos_selection = ['torso_link']
        teleop_body_ang_vel_selection = ['torso_link']
        penalty_reward_names = [  "torques",
                                "torque_limits",
                                "dof_acc",
                                "dof_vel",
                                # action_rate : -0.6 # -0.6  # -0.3 # -0.3 -0.12 -0.01
                                "lower_action_rate",
                                "upper_action_rate",
                                "dof_pos_limits",
                                "termination",
                                "feet_contact_forces",
                                "stumble",
                                "feet_air_time_teleop",
                                "slippage",
                                "feet_ori",
                                "orientation",
                                "in_the_air",
                                "stable_lower_when_vrclose"]
    
    class normalization:
        class obs_scales: # no normalization for nows
            lin_vel = 1.0 # 2.0
            lin_acc = 1.0 # ????????????
            ang_vel = 1.0 # 0.25
            dof_pos = 1.0 # 1.0
            dof_vel = 1.0 # 0.05
            height_measurements = 1.0 # 5.0
            body_pos = 1.0
            body_lin_vel = 1.0
            body_rot = 1.0
            delta_base_pos = 1.0
            delta_heading = 1.0
        clip_observations = 100.
        clip_actions = 100.

    class amp():
        num_obs_steps = 10
        num_obs_per_step = 20 + 3 # 19 joint angles + 3 base ang vel

class StompyProTeleopCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.005

    class amp():
        amp_input_dim = StompyProTeleopCfg.amp.num_obs_steps * StompyProTeleopCfg.amp.num_obs_per_step
        amp_disc_hidden_dims = [512, 256]

        amp_replay_buffer_size = 10000
        amp_demo_buffer_size = 10000
        amp_demo_fetch_batch_size = 512
        amp_learning_rate = 1.e-4

        amp_reward_coef = 2.0