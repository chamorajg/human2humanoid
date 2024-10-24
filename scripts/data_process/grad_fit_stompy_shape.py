import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from phc.utils import torch_utils
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser, 
)

import joblib
import torch
import torch.nn.functional as F
import math
from phc.utils.pytorch3d_transforms import axis_angle_to_matrix
from torch.autograd import Variable
from scipy.ndimage import gaussian_filter1d
from tqdm.notebook import tqdm
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPLH_MUJOCO_NAMES
from phc.utils.torch_stompy_humanoid_batch import Stompy_Batch, STOMPY_ROTATION_AXIS
stompy_joint_names = [
                    #   'torso_link',
                      'L_buttock', 'L_leg', 'L_thigh', 'L_calf', 'L_foot', 
                      'L_clav', 'L_scapula', 'L_uarm', 'L_farm', 
                      'R_buttock', 'R_leg', 'R_thigh', 'R_calf', 'R_foot',
                      'R_clav', 'R_scapula', 'R_uarm', 'R_farm']


stompy_fk = Stompy_Batch(extend_head=True) # load forward kinematics model
#### Define corresonpdances between h1 and smpl joints
stompy_joint_names_augment = stompy_joint_names + ["left_hand_keypoint_link", "right_hand_keypoint_link", "head_link"]
stompy_joint_pick = ['L_thigh', "L_calf", "L_foot", "L_scapula", "L_uarm", "L_farm",  'R_thigh', 'R_calf', 'R_foot', "R_scapula", "R_uarm", "R_farm", "left_hand_keypoint_link", "right_hand_keypoint_link", "head_link"]
smpl_joint_pick = ["L_Hip",  "L_Knee", "L_Ankle", "L_Shoulder", "L_Elbow",  "L_Wrist", "R_Hip", "R_Knee", "R_Ankle", "R_Shoulder", "R_Elbow", "R_Wrist", "L_Hand",  "R_Hand", "Head"]
stompy_joint_pick_idx = [ stompy_joint_names_augment.index(j) for j in stompy_joint_pick]
smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]


#### Preparing fitting varialbes
device = torch.device("cpu")
pose_aa_stompy = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None, ], 21, axis = 2), 1, axis = 1)
pose_aa_stompy = torch.from_numpy(pose_aa_stompy).float()

dof_pos = torch.zeros((1, 18))
pose_aa_stompy = torch.cat([torch.zeros((1, 1, 3)), STOMPY_ROTATION_AXIS * dof_pos[..., None], torch.zeros((1, 2, 3))], axis = 1)

root_trans = torch.zeros((1, 1, 3))    

###### prepare SMPL default pause for H1
pose_aa_stand = np.zeros((1, 72))
rotvec = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).as_rotvec()
pose_aa_stand[:, :3] = rotvec
pose_aa_stand = pose_aa_stand.reshape(-1, 24, 3)
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Shoulder')] = sRot.from_euler("xyz", [0, 0, -np.pi/2],  degrees = False).as_rotvec()
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Shoulder')] = sRot.from_euler("xyz", [0, 0, np.pi/2],  degrees = False).as_rotvec()
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('L_Elbow')] = sRot.from_euler("xyz", [0, -np.pi/2, 0],  degrees = False).as_rotvec()
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index('R_Elbow')] = sRot.from_euler("xyz", [0, np.pi/2, 0],  degrees = False).as_rotvec()
pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72))

smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")

###### Shape fitting
trans = torch.zeros([1, 3])
beta = torch.zeros([1, 10])
verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, beta , trans)
offset = joints[:, 0] - trans
root_trans_offset = trans + offset

fk_return = stompy_fk.fk_batch(pose_aa_stompy[None, ], root_trans_offset[None, 0:1])

shape_new = Variable(torch.zeros([1, 10]).to(device), requires_grad=True)
scale = Variable(torch.ones([1]).to(device), requires_grad=True)
optimizer_shape = torch.optim.Adam([shape_new, scale],lr=0.1)


for iteration in range(1000):
    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, shape_new, trans[0:1])
    root_pos = joints[:, 0]
    joints = (joints - joints[:, 0]) * scale + root_pos
    diff = fk_return.global_translation_extend[:, :, stompy_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
    loss_g = diff.norm(dim = -1).mean() 
    loss = loss_g
    if iteration % 100 == 0:
        print(iteration, loss.item() * 1000)

    optimizer_shape.zero_grad()
    loss.backward()
    optimizer_shape.step()


joblib.dump((shape_new.detach().cpu(), scale.detach().cpu()), "data/stompy/shape_optimized_v1.pkl") # V2 has hip jointsrea