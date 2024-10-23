import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "poselib"))

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from phc.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
    SMPL_BONE_ORDER_NAMES, 
)
import joblib
from phc.utils.rotation_conversions import axis_angle_to_matrix
from phc.utils.torch_stompy_humanoid_batch import Stompy_Batch
from torch.autograd import Variable
from tqdm import tqdm

device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

stompy_rotation_axis = torch.tensor([
        # [0, 0, 1], # torso

       [ 0, 1, 0], # L_hip_y # pitch
       [ 1, 0, 0], # L_hip_x # roll
       [ 0, 0, 1], # L_hip_z # yaw
       
       [ 0, 1, 0], # L_knee
       [ 0, 1, 0], # L_ankle_y # pitch

       [ 0, 1, 0], # L_shoulder_y # pitch
       [ 1, 0, 0], # L_shoulder_x # roll
       [ 0, 0, 1], # L_shoulder_z # yaw
       [ 1, 0, 0], # L_elbow_x # roll
        
       
       [ 0, 1, 0], # R_hip_y # pitch
       [ 1, 0, 0], # R_hip_x # roll
       [ 0, 0, 1], # R_hip_z # yaw
       
       [ 0, 1, 0], # R_knee
       [ 0, 1, 0], # R_ankle_y # pitch

    #    [], # left_hand_keypoint_joint
       [ 0, 1, 0], # R_shoulder_y # pitch
       [ 1, 0, 0], # R_shoulder_x # roll
       [ 0, 0, 1], # R_shoulder_z # yaw
       [ 1, 0, 0], # R_elbow_x # roll
    #    [], # right_hand_keypoint_joint
       ],).to(device)

stompy_joint_names = [
                    #   'torso_link',
                      'L_buttock', 'L_leg', 'L_thigh', 'L_calf', 'L_foot', 
                      'R_buttock', 'R_leg', 'R_thigh', 'R_calf', 'R_foot',
                      'L_clav', 'L_scapula', 'L_uarm', 'L_farm', 
                      'R_clav', 'R_scapula', 'R_uarm', 'R_farm']


#### Define corresonpdances between h1 and smpl joints
stompy_joint_names_augment = stompy_joint_names + ["left_hand_keypoint_link", "right_hand_keypoint_link", "head_link"]
stompy_joint_pick = ['L_thigh', "L_calf", "L_foot",  'R_thigh', 'R_calf', 'R_foot', "L_scapula", "L_farm", "left_hand_keypoint_link", "R_scapula", "R_farm", "right_hand_keypoint_link"]
smpl_joint_pick = ["L_Hip",  "L_Knee", "L_Ankle",  "R_Hip", "R_Knee", "R_Ankle", "L_Shoulder", "L_Elbow", "L_Hand", "R_Shoulder", "R_Elbow", "R_Hand"]
stompy_joint_pick_idx = [ stompy_joint_names_augment.index(j) for j in stompy_joint_pick]
smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]


smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")
smpl_parser_n.to(device)

amass_data = joblib.load('/home/kasm-user/PHC/sample_data/amass_copycat_take6_train.pkl') # From PHC
shape_new, scale = joblib.load("data/stompy/shape_optimized_v1.pkl")
shape_new = shape_new.to(device)



stompy_fk = Stompy_Batch(device = device)
data_dump = {}
pbar = tqdm(amass_data.keys())
for data_key in pbar:
    trans = torch.from_numpy(amass_data[data_key]['trans']).float().to(device)
    N = trans.shape[0]
    pose_aa_walk = torch.from_numpy(np.concatenate((amass_data[data_key]['pose_aa'][:, :66], np.zeros((N, 6))), axis = -1)).float().to(device)


    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, torch.zeros((1, 10)).to(device), trans)
    offset = joints[:, 0] - trans
    root_trans_offset = trans + offset


    pose_aa_stompy = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None, ], 21, axis = 2), N, axis = 1)
    pose_aa_stompy[..., 0, :] = (sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()
    pose_aa_stompy = torch.from_numpy(pose_aa_stompy).float().to(device)
    gt_root_rot = torch.from_numpy((sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()).float().to(device)

    dof_pos = torch.zeros((1, N, 18, 1)).to(device)

    dof_pos_new = Variable(dof_pos, requires_grad=True)
    optimizer_pose = torch.optim.Adadelta([dof_pos_new],lr=100)

    for iteration in range(500):
        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
        pose_aa_stompy_new = torch.cat([gt_root_rot[None, :, None], stompy_rotation_axis * dof_pos_new, torch.zeros((1, N, 2, 3)).to(device)], axis = 2).to(device)
        fk_return = stompy_fk.fk_batch(pose_aa_stompy_new, root_trans_offset[None, ])
        diff = fk_return['global_translation'][:, :, stompy_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
        loss_g = diff.norm(dim = -1).mean() 
        loss = loss_g
        
        if iteration % 50 == 0:
            pbar.set_description_str(f"{iteration} {loss.item() * 1000}")

        optimizer_pose.zero_grad()
        loss.backward()
        optimizer_pose.step()
        dof_pos_new.data.clamp_(stompy_fk.joints_range[:, 0, None], stompy_fk.joints_range[:, 1, None])
        
    dof_pos_new.data.clamp_(stompy_fk.joints_range[:, 0, None], stompy_fk.joints_range[:, 1, None])
    pose_aa_stompy_new = torch.cat([gt_root_rot[None, :, None], stompy_rotation_axis * dof_pos_new, torch.zeros((1, N, 2, 3)).to(device)], axis = 2)
    fk_return = stompy_fk.fk_batch(pose_aa_stompy_new, root_trans_offset[None, ])

    root_trans_offset_dump = root_trans_offset.clone()

    root_trans_offset_dump[..., 2] -= fk_return.global_translation[..., 2].min().item() - 0.08

    print(dof_pos_new.shape)
    data_dump[data_key]={
            "root_trans_offset": root_trans_offset_dump.squeeze().cpu().detach().numpy(),
            "pose_aa": pose_aa_stompy_new.squeeze().cpu().detach().numpy(),   
            "dof": dof_pos_new.squeeze().detach().cpu().numpy(), 
            "root_rot": sRot.from_rotvec(gt_root_rot.cpu().numpy()).as_quat(),
            }

import ipdb; ipdb.set_trace()
joblib.dump(data_dump, "data/stompy/amass_train.pkl")