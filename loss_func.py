'''
 @FileName    : loss_func.py
 @EditTime    : 2022-01-13 19:16:39
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : define loss functions here
'''

import torch.nn as nn
import torch
import numpy as np
from utils.geometry import batch_rodrigues

import time
from utils.mesh_intersection.bvh_search_tree import BVH
import utils.mesh_intersection.loss as collisions_loss
from utils.mesh_intersection.filter_faces import FilterFaces
from utils.FileLoaders import load_pkl
from utils.geometry import perspective_projection
import cv2

# class L1(nn.Module):
#     def __init__(self, device):
#         super(L1, self).__init__()
#         self.device = device
#         self.L1Loss = nn.L1Loss()

#     def forward(self, x, y):
#         diff = self.L1Loss(x, y)
#         # diff = diff / b
#         return diff


# class KL_Loss(nn.Module):
#     def __init__(self, device):
#         super(KL_Loss, self).__init__()
#         self.device = device
#         self.kl_coef = 0.005

#     def forward(self, q_z):
#         b = q_z.mean.shape[0]
#         loss_dict = {}

#         # KL loss
#         p_z = torch.distributions.normal.Normal(
#             loc=torch.zeros_like(q_z.loc, requires_grad=False).to(q_z.loc.device).type(q_z.loc.dtype),
#             scale=torch.ones_like(q_z.scale, requires_grad=False).to(q_z.scale.device).type(q_z.scale.dtype))
#         loss_kl = torch.distributions.kl.kl_divergence(q_z, p_z)

#         loss_kl = loss_kl.sum()
#         loss_kl = loss_kl / b
#         loss_dict['loss_kl'] = self.kl_coef * loss_kl
#         return loss_dict

class SMPL_Loss(nn.Module):
    def __init__(self, device):
        super(SMPL_Loss, self).__init__()
        self.device = device
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.beta_loss_weight = 1.0
        self.pose_loss_weight = 1000.0
        self.trans_loss_weight = 1000.0

    def forward(self, pred_rotmat, gt_pose, pred_betas, gt_betas, pred_trans, gt_trans, has_smpl, valid):
        loss_dict = {}

        # pred_rotmat = pred_rotmat[valid == 1]
        # gt_pose = gt_pose[valid == 1]
        # pred_betas = pred_betas[valid == 1]
        # gt_betas = gt_betas[valid == 1]
        # pred_trans = pred_trans[valid == 1]
        # gt_trans = gt_trans[valid == 1]
        # has_smpl = has_smpl[valid == 1]

        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl == 1]

        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        
        pred_trans_valid = pred_trans[has_smpl == 1]
        gt_trans_valid = pred_trans[has_smpl == 1]

        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
            loss_regr_trans = self.criterion_regr(pred_trans, gt_trans)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)[0]
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)[0]
            loss_regr_trans = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        loss_dict['pose_Loss'] = loss_regr_pose * self.pose_loss_weight
        loss_dict['shape_Loss'] = loss_regr_betas * self.beta_loss_weight
        loss_dict['trans_Loss'] = loss_regr_trans * self.trans_loss_weight
        return loss_dict


class Keyp_Loss(nn.Module):
    def __init__(self, device):
        super(Keyp_Loss, self).__init__()
        self.device = device
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.keyp_weight = 10000.0
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]
        self.halpe2coco = [0,18,6,8,10,5,7,9,12,14,16,11,13,15,2,1,4,3]

    def forward(self, pred_keypoints_2d, gt_keypoints_2d, valid):
        loss_dict = {}
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        # pred_keypoints_2d = pred_keypoints_2d[valid == 1]
        # gt_keypoints_2d = gt_keypoints_2d[valid == 1]
        
        # if gt_keypoints_2d.shape[-2] == 18:
        #     pred_keypoints_2d = pred_keypoints_2d[:,self.halpe2coco]
        #     # gt_keypoints_2d = gt_keypoints_2d[:,self.halpe2coco]
        # else:
        #     pred_keypoints_2d = pred_keypoints_2d[:,self.halpe2lsp]
        #     gt_keypoints_2d = gt_keypoints_2d[:,self.halpe2lsp]
        
        pred_keypoints_2d = pred_keypoints_2d[:,self.halpe2lsp]
        gt_keypoints_2d = gt_keypoints_2d[:,self.halpe2lsp]

        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()

        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()

        if loss > 300 or loss != loss:
            loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        loss_dict['Keyp_Loss'] = loss * self.keyp_weight
        return loss_dict

class Mesh_Loss(nn.Module):
    def __init__(self, device):
        super(Mesh_Loss, self).__init__()
        self.device = device
        # self.criterion_vert = nn.L1Loss().to(self.device)
        self.criterion_vert = nn.MSELoss().to(self.device)
        self.verts_weight = 5000.0

    def forward(self, pred_vertices, gt_vertices, has_smpl, valid):
        loss_dict = {}
        # pred_vertices = pred_vertices[valid == 1]
        # gt_vertices = gt_vertices[valid == 1]
        # has_smpl = has_smpl[valid == 1]

        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]

        if len(gt_vertices_with_shape) > 0:
            vert_loss = self.criterion_vert(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            vert_loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        loss_dict['Mesh_Loss'] = vert_loss * self.verts_weight
        return loss_dict

# class Skeleton_Loss(nn.Module):
#     def __init__(self, device):
#         super(Skeleton_Loss, self).__init__()
#         self.device = device
#         self.criterion_vert = nn.L1Loss().to(self.device)
#         self.criterion_joint = nn.MSELoss(reduction='none').to(self.device)
#         self.skeleton_weight = 5.0
#         self.verts_weight = 5.0
#         self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]
#         self.right_start = [12, 8, 7, 12, 2, 1]
#         self.right_end = [8, 7, 6, 2, 1, 0]
#         self.left_start = [12, 9, 10, 12, 3, 4]
#         self.left_end = [9, 10, 11, 3, 4, 5]

#     def forward(self, pred_joints):
#         loss_dict = {}
        
#         pred_joints = pred_joints[:,self.halpe2lsp]
        
#         left_bone_length = torch.norm(pred_joints[:, self.left_start] - pred_joints[:, self.left_end], dim=-1)
#         right_bone_length = torch.norm(pred_joints[:, self.right_start] - pred_joints[:, self.right_end], dim=-1)

#         skeleton_loss = self.criterion_joint(left_bone_length, right_bone_length).mean()

#         loss_dict['skeleton_loss'] = skeleton_loss * self.skeleton_weight
#         return loss_dict

# class Joint_Loss(nn.Module):
#     def __init__(self, device):
#         super(Joint_Loss, self).__init__()
#         self.device = device
#         self.criterion_vert = nn.L1Loss().to(self.device)
#         self.criterion_joint = nn.MSELoss(reduction='none').to(self.device)
#         self.joint_weight = 100.0
#         self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

#     def forward(self, pred_joints, gt_joints, has_3d, valid):
#         loss_dict = {}
        
#         pred_joints = pred_joints[valid == 1]
#         gt_joints = gt_joints[valid == 1]
#         has_3d = has_3d[valid == 1]

#         pred_joints = pred_joints[:,self.halpe2lsp]
#         gt_joints = gt_joints[:,self.halpe2lsp]

#         conf = gt_joints[:, :, -1].unsqueeze(-1).clone()[has_3d == 1]

#         gt_pelvis = (gt_joints[:,2,:3] + gt_joints[:,3,:3]) / 2.
#         gt_joints[:,:,:-1] = gt_joints[:,:,:-1] - gt_pelvis[:,None,:]

#         pred_pelvis = (pred_joints[:,2,:] + pred_joints[:,3,:]) / 2.
#         pred_joints = pred_joints - pred_pelvis[:,None,:]

#         gt_joints = gt_joints[has_3d == 1]
#         pred_joints = pred_joints[has_3d == 1]

#         if len(gt_joints) > 0:
#             joint_loss = (conf * self.criterion_joint(pred_joints, gt_joints[:, :, :-1])).mean()
#         else:
#             joint_loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

#         loss_dict['joint_loss'] = joint_loss * self.joint_weight
#         return loss_dict

class Joint_Loss(nn.Module):
    def __init__(self, device):
        super(Joint_Loss, self).__init__()
        self.device = device
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]
        self.criterion_joint = nn.MSELoss(reduction='none').to(self.device)
        self.mpjpe_weight = 5000.0
        self.pa_mpjpe_weight = 5000.0
        self.mpjpe_wrt_one_weight = 5000.0
        self.dpa_mpjpe_weight = 1.0

    def forward(self, pred_joints, pred_trans, gt_joints, gt_trans, valid, dshape):
        loss_dict = {}

        g_joint_loss = self.forward_g_joints(pred_joints, pred_trans, gt_joints, gt_trans, valid)
        loss_dict['g_joint_loss'] = g_joint_loss * self.mpjpe_weight # in camera coordinate
        
        ra_joints_loss = self.forward_ra_joints(pred_joints, gt_joints, valid)
        loss_dict['ra_joints_loss'] = ra_joints_loss * self.pa_mpjpe_weight # aling by pelvis
        
        joint_loss_wrt_one = self.forward_joints_wrt_one(pred_joints, pred_trans, gt_joints, gt_trans, dshape, valid)
        loss_dict['joint_loss_wrt_one'] = joint_loss_wrt_one * self.mpjpe_wrt_one_weight # align by the first person's pelvis
              
        dpa_joints_loss = self.foward_dpa_joints(pred_joints, pred_trans, gt_joints, gt_trans, dshape, valid)
        loss_dict['dpa_joints_loss'] = dpa_joints_loss * self.dpa_mpjpe_weight # depth-aware joints loss
        
        return loss_dict
    
    def forward_g_joints(self, pred_joints, pred_trans, gt_joints, gt_trans, valid):      
        pred_joints = pred_joints + pred_trans[:,None,:]
        gt_joints = gt_joints[...,:3] + gt_trans[:,None,:]

        # pred_joints = pred_joints[valid == 1]
        # gt_joints = gt_joints[valid == 1]
        gt_joints = gt_joints[...,:3]

        loss = self.criterion_joint(pred_joints, gt_joints).mean()
        
        return loss
    
    def forward_joints_wrt_one(self, pred_joints, pred_trans, gt_joints, gt_trans, dshape, valid):

        # joints in camera coordinate
        pred_joints = pred_joints + pred_trans[:,None,:]
        gt_joints = gt_joints[...,:3] + gt_trans[:,None,:]

        # pred_joints = pred_joints[valid == 1]
        # gt_joints = gt_joints[valid == 1]
        
        pred_joints = pred_joints.reshape(dshape[0], dshape[1], dshape[2], -1, 3) # (bs, frame_length, num_people, 26, 3)
        gt_joints = gt_joints.reshape(dshape[0], dshape[1], dshape[2], -1, 3)
        
        pred_joints_a = pred_joints[:,:,0] # (bs, frame_length, 26, 3)
        gt_joints_a = gt_joints[:,:,0]
        
        # Align by pelvis of the first person
        pred_pelvis_a = pred_joints_a[:,:,19]
        gt_pelvis_a = gt_joints_a[:,:,19]
        
        pred_joints = pred_joints - pred_pelvis_a[:,:,None,None,:]
        gt_joints = gt_joints - gt_pelvis_a[:,:,None,None,:]
        
        loss = self.criterion_joint(pred_joints, gt_joints).mean()
        
        return loss

    def forward_ra_joints(self, pred_joints, gt_joints, valid):
        # pred_joints = pred_joints[valid == 1]
        # gt_joints = gt_joints[valid == 1]

        gt_pelvis = gt_joints[...,19,:3]
        gt_joints = gt_joints[...,:3] - gt_pelvis[...,None,:]

        pred_pelvis = pred_joints[...,19,:3]
        pred_joints = pred_joints - pred_pelvis[:,None,:]

        loss = self.criterion_joint(pred_joints, gt_joints).mean()
        
        return loss
    
    def foward_dpa_joints(self, pred_joints, pred_trans, gt_joints, gt_trans, dshape, valid):
        """
        depth-aware joints loss
        """
        # get the joints in camera coordinate
        pred_joints = pred_joints + pred_trans[:,None,:]
        gt_joints = gt_joints[...,:3] + gt_trans[:,None,:]

        # pred_joints = pred_joints[valid == 1]
        # gt_joints = gt_joints[valid == 1]
        
        # reshape the joints
        pred_joints = pred_joints.reshape(dshape[0]*dshape[1], dshape[2]*26, 3) # (bs*frame_length, num_people, 26, 3)
        gt_joints = gt_joints.reshape(dshape[0]*dshape[1], dshape[2]*26, 3)
        
        # get the depth order of joints
        pred_order = self.calculate_depth_order(pred_joints)
        gt_order = self.calculate_depth_order(gt_joints)
        
        # calculate depth order loss
        depth_order_loss = torch.log(1+torch.exp(pred_order - gt_order)).mean()
        return depth_order_loss                
        
    def calculate_depth_order(self, joints):
        """
        Calculate the depth order of joints based on the z-axis values.
        Args:
            joints: (bs, num_joints, 3) tensor representing the joints.
        Returns:
            depth_order: (bs, num_joints) tensor representing the depth order of joints.
        """
        depth_order = torch.argsort(joints[..., 2], dim=-1)
        return depth_order

    def batch_compute_similarity_transform(self, S1, S2):
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        '''
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.permute(0,2,1)
            S2 = S2.permute(0,2,1)
            transposed = True
        assert(S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=-1, keepdims=True)
        mu2 = S2.mean(axis=-1, keepdims=True)

        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = torch.sum(X1**2, dim=1).sum(dim=1)

        # 3. The outer product of X1 and X2.
        K = X1.bmm(X2.permute(0,2,1))

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, V = torch.svd(K)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
        Z = Z.repeat(U.shape[0],1,1)
        t1 = U.bmm(V.permute(0,2,1))
        t2 = torch.det(t1)
        Z[:,-1, -1] = Z[:,-1, -1] * torch.sign(t2)
        # Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

        # Construct R.
        R = V.bmm(Z.bmm(U.permute(0,2,1)))

        # 5. Recover scale.
        scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

        # 6. Recover translation.
        t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

        # 7. Error:
        S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

        if transposed:
            S1_hat = S1_hat.permute(0,2,1)

        return S1_hat

class Int_Loss(nn.Module):
    def __init__(self, device):
        super(Int_Loss, self).__init__()
        self.device = device
        self.criterion_dm = nn.MSELoss().to(self.device)
        self.interaction_weight = 1000.0
        self.num_people = 2
        self.eval_mode = False
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]


    def forward(self, pred_joints, pred_trans, gt_joints, gt_trans, dshape, valid):
        loss_dict = {}
        
        bs, frame_length, num_people = dshape[0], dshape[1], dshape[2]

        pred_joints = pred_joints + pred_trans[:,None,:]
        gt_joints = gt_joints[...,:3] + gt_trans[:,None,:]

        pred_joints = pred_joints[...,self.halpe2lsp,:] * valid[:,None,None]
        gt_joints = gt_joints[...,self.halpe2lsp,:] * valid[:,None,None]
        
        joint_num = pred_joints.shape[-2]

        pred_joints = pred_joints.reshape(bs, frame_length, num_people, joint_num, 3) # [bs, frame_length, num_people, joint_num, 3]
        gt_joints = gt_joints.reshape(bs, frame_length, num_people, joint_num, 3)
        
        
        pred_joints_a = pred_joints[:,:,0] # [bs, frame_length, joint_num, 3]
        pred_joints_b = pred_joints[:,:,1]
        gt_joints_a = gt_joints[:,:,0]
        gt_joints_b = gt_joints[:,:,1]
        
        # define distance map as every a joint to every b joint, an 26x26 matrix
        pred_joints_distance_matrix = torch.cdist(pred_joints_a, pred_joints_b, p=2) # [bs, frame_length, joint_num, joint_num]
        gt_joints_distance_matrix = torch.cdist(gt_joints_a, gt_joints_b, p=2)
        
        # calculate the loss
        # apply this loss only when the element in the distance map is less than 1
        if self.eval_mode:
            mask = torch.ones_like(gt_joints_distance_matrix).float() # [bs, frame_length, joint_num, joint_num]
        else:
            mask = (gt_joints_distance_matrix < 1).float() # [bs, frame_length, joint_num, joint_num]

        # print(mask.sum())
        int_loss = torch.abs(pred_joints_distance_matrix-gt_joints_distance_matrix) * mask # [bs, frame_length, joint_num, joint_num]
        int_loss = int_loss.sum(dim=[-1,-2]) # [bs, frame_length]
        int_loss = int_loss / (joint_num*joint_num) # [bs, frame_length]
        if self.eval_mode:
            int_loss = int_loss.sum()
        else:
            int_loss = int_loss.sum()*2/valid.sum()

        loss_dict['Int_Loss'] = int_loss * self.interaction_weight

        return loss_dict
    
    def set_eval_mode(self):
        self.interaction_weight = 1000.0
        self.eval_mode = True


class Interaction(nn.Module):
    def __init__(self, device):
        super(Interaction, self).__init__()
        self.device = device
        self.criterion_vert = nn.L1Loss().to(self.device)
        self.interaction_weight = 1000.0
        self.num_people = 2
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

    def forward(self, pred_joints, pred_trans, gt_joints, gt_trans, dshape, valid):
        loss_dict = {}
        
        bs, frame_length, num_people = dshape[0], dshape[1], dshape[2]
        pred_joints[...,:3] = pred_joints[...,:3] + pred_trans[:,None,:]
        gt_joints[...,:3] = gt_joints[...,:3] + gt_trans[:,None,:]
        
        pred_joints = pred_joints * valid[:,None,None]
        gt_joints = gt_joints * valid[:,None,None]

        pred_joints = pred_joints[:,self.halpe2lsp]
        gt_joints = gt_joints[:,self.halpe2lsp]

        gt_joints = gt_joints[...,:3]
        gt_joints = gt_joints.reshape(bs, frame_length, num_people, len(self.halpe2lsp), 3) # bs*frame_length, num_people, len(self.halpe2lsp), 3
        pred_joints = pred_joints.reshape(bs, frame_length, num_people, len(self.halpe2lsp), 3)

        gt_joint_a, gt_joint_b = gt_joints[:,:,0], gt_joints[:,:,1] # bs, frame_length, len(self.halpe2lsp), 3
        gt_interaction = torch.norm(gt_joint_a - gt_joint_b, dim=-1) # bs, frame_length, len(self.halpe2lsp)

        pred_joint_a, pred_joint_b = pred_joints[:,:,0], pred_joints[:,:,1]
        pred_interaction = torch.norm(pred_joint_a - pred_joint_b, dim=-1)

        interaction = torch.abs(gt_interaction - pred_interaction)
                
        interaction = interaction.mean(dim=[-1]) # bs, frame_length
        interaction = interaction.sum() *2.
        loss_dict['Interaction'] = interaction * self.interaction_weight

        return loss_dict

class Vel_Loss(nn.Module):
    def __init__(self, device):
        super(Vel_Loss, self).__init__()
        self.device = device
        self.criterion_joint_vel = nn.MSELoss(reduction='none').to(self.device)
        self.vel_weight = 5000.0
        self.eval_mode = False
        
    def forward(self, pred_joints, pred_transl, gt_joints, gt_transl, dshape, valid):
        loss_dict = {}
        
        pred_joints = pred_joints + pred_transl[:,None,:]
        gt_joints = gt_joints[...,:3] + gt_transl[:,None,:]

        pred_joints = pred_joints.reshape(dshape[0], dshape[1], dshape[2], -1, 3)
        gt_joints = gt_joints.reshape(dshape[0], dshape[1], dshape[2], -1, 3)
        valid = valid.reshape(dshape[0], dshape[1], dshape[2])

        pred_vel = pred_joints[:,1:] - pred_joints[:,:1]
        gt_vel = gt_joints[:,1:] - gt_joints[:,:1]

        pred_acc = pred_joints[:,2:] - pred_joints[:,:1]
        gt_acc = gt_joints[:,2:] - gt_joints[:,:1]
        
        vel_loss = (self.criterion_joint_vel(pred_vel, gt_vel)).mean(dim=[-1,-2]) # [bs, frame_length, num_people]
        vel_loss = vel_loss * valid[:,:-1]
        acc_loss = (self.criterion_joint_vel(pred_acc, gt_acc)).mean(dim=[-1,-2]) # [bs, frame_length, num_people]
        acc_loss = acc_loss * valid[:,:-2]
        if self.eval_mode:
            vel_loss = vel_loss.sum()
            acc_loss = acc_loss.sum()
        else:
            vel_loss = vel_loss.sum() / valid.sum()
            acc_loss = acc_loss.sum() / valid.sum()


        loss_dict['Vel_Loss'] = vel_loss * self.vel_weight
        loss_dict['Acc_Loss'] = acc_loss * self.vel_weight
        return loss_dict
    
    def forward_instance(self, pred_joints, pred_transl, gt_joints, gt_transl, dshape, has_3d, valid):
                
        pred_joints = pred_joints + pred_transl[:,None,:]
        gt_joints = gt_joints[...,:3] + gt_transl[:,None,:]

        pred_joints = pred_joints.reshape(dshape[0], dshape[1], dshape[2], -1, 3)
        gt_joints = gt_joints.reshape(dshape[0], dshape[1], dshape[2], -1, 3)
        valid = valid.reshape(dshape[0], dshape[1], dshape[2])

        pred_vel = pred_joints[:,1:] - pred_joints[:,:1]
        gt_vel = gt_joints[:,1:] - gt_joints[:,:1]

        vel_loss_frame = self.criterion_joint_vel(pred_vel, gt_vel).mean(dim=-1) # [bs, frame_length, num_people]
        vel_loss_frame = vel_loss_frame * valid
        if self.eval_mode:
            vel_loss_frame = vel_loss_frame.sum()
        else:
            vel_loss_frame = vel_loss_frame.sum() / valid.sum()
        
        return vel_loss_frame
    
    def set_eval_mode(self):
        self.vel_weight = 1000.0
        self.acc_weight = 1000.0
        self.eval_mode = True

# class Joint_abs_Loss(nn.Module):
#     def __init__(self, device):
#         super(Joint_abs_Loss, self).__init__()
#         self.device = device
#         self.criterion_vert = nn.L1Loss().to(self.device)
#         self.criterion_joint = nn.MSELoss(reduction='none').to(self.device)
#         self.joint_weight = 0.5
#         self.verts_weight = 5.0
#         self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

#     def forward(self, pred_joints, gt_joints, has_3d):
#         loss_dict = {}
        
#         pred_joints = pred_joints[:,self.halpe2lsp]
#         gt_joints = gt_joints[:,self.halpe2lsp]

#         conf = gt_joints[:, :, -1].unsqueeze(-1).clone()[has_3d == 1]

#         gt_joints = gt_joints[has_3d == 1]
#         pred_joints = pred_joints[has_3d == 1]

#         if len(gt_joints) > 0:
#             joint_loss = (conf * self.criterion_joint(pred_joints, gt_joints[:, :, :-1])).mean()
#         else:
#             joint_loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

#         loss_dict['joint_abs_loss'] = joint_loss * self.joint_weight
#         return loss_dict

# class Latent_Diff(nn.Module):
#     def __init__(self, device):
#         super(Latent_Diff, self).__init__()
#         self.device = device
#         self.weight = 0.02

#     def forward(self, diff):
#         loss_dict = {}

#         loss_dict['latent_diff'] = diff.sum() * self.weight
        
#         return loss_dict

class Pen_Loss(nn.Module):
    def __init__(self, device, smpl):
        super(Pen_Loss, self).__init__()
        self.device = device
        self.weight = 50
        self.smpl = smpl

        self.search_tree = BVH(max_collisions=8)
        self.pen_distance = collisions_loss.DistanceFieldPenetrationLoss(sigma=0.0001,
                                                         point2plane=False,
                                                         vectorized=True)

        self.part_segm_fn = False #"data/smpl_segmentation.pkl"
        if self.part_segm_fn:
            data = load_pkl(self.part_segm_fn)

            faces_segm = data['segm']
            ign_part_pairs = [
                "9,16", "9,17", "6,16", "6,17", "1,2",
                "33,40", "33,41", "30,40", "30,41", "24,25",
            ]

            faces_segm = torch.tensor(faces_segm, dtype=torch.long,
                                device=self.device).unsqueeze_(0).repeat([2, 1]) # (2, 13766)

            faces_segm = faces_segm + \
                (torch.arange(2, dtype=torch.long).to(self.device) * 24)[:, None]
            faces_segm = faces_segm.reshape(-1) # (2*13766, )

            # Create the module used to filter invalid collision pairs
            self.filter_faces = FilterFaces(faces_segm=faces_segm, ign_part_pairs=ign_part_pairs).to(device=self.device)

    def forward(self, verts, trans):
        loss_dict = {}

        vertices = verts + trans[:,None,:]
        face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long,
                                device=vertices.device).unsqueeze_(0).repeat([vertices.shape[0],
                                                                        1, 1])
        bs, nv = vertices.shape[:2] # nv: 6890
        bs, nf = face_tensor.shape[:2] # nf: 13776
        faces_idx = face_tensor + (torch.arange(bs, dtype=torch.long).to(vertices.device) * nv)[:, None, None]
        faces_idx = faces_idx.reshape(bs // 2, -1, 3)
        triangles = vertices.view([-1, 3])[faces_idx]

        print_timings = False
        if print_timings:
            start = time.time()
        collision_idxs = self.search_tree(triangles) # (128, n_coll_pairs, 2)
        if print_timings:
            torch.cuda.synchronize()
            print('Collision Detection: {:5f} ms'.format((time.time() - start) * 1000))

        if self.part_segm_fn:
            if print_timings:
                start = time.time()
            collision_idxs = self.filter_faces(collision_idxs)
            if print_timings:
                torch.cuda.synchronize()
                print('Collision filtering: {:5f}ms'.format((time.time() -
                                                            start) * 1000))

        if print_timings:
                start = time.time()
        pen_loss = self.pen_distance(triangles, collision_idxs)
        if print_timings:
            torch.cuda.synchronize()
            print('Penetration loss: {:5f} ms'.format((time.time() - start) * 1000))

        pen_loss = pen_loss[pen_loss<2000]
        
        if len(pen_loss) > 0:
            pen_loss = torch.sigmoid(pen_loss / 2000.) - 0.5
            loss_dict['Pen_Loss'] = pen_loss.mean() * self.weight
        else:
            loss_dict['Pen_Loss'] = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        return loss_dict

# class Plane_Loss(nn.Module):
#     '''
    
#     '''
#     def __init__(self, device):
#         super(Plane_Loss, self).__init__()
#         self.device = device
#         self.criterion_vert = nn.L1Loss().to(self.device)
#         self.criterion_joint = nn.MSELoss(reduction='none').to(self.device)
#         self.height_weight = 1

#     def forward(self, pred_joints, valids):
#         loss_dict = {}
#         batchsize = len(valids)

#         idx = 0
#         loss = 0.
#         for img in valids.detach().to(torch.int8):
#             num = img.sum()

#             if num <= 1:
#                 dis_std = torch.FloatTensor(1).fill_(0.).to(self.device)[0]
#             else:
#                 joints = pred_joints[idx:idx+num]

#                 bottom = (joints[:,15] + joints[:,16]) / 2
#                 top = joints[:,17]

#                 l = (top - bottom) / torch.norm(top - bottom, dim=1)[:,None]
#                 norm = torch.mean(l, dim=0)

#                 root = (joints[:,11] + joints[:,12]) / 2 #joints[:,19]

#                 proj = torch.matmul(root, norm)

#                 dis_std = proj.std()

#             idx += num
#             loss += dis_std

#         loss_dict['plane_loss'] = loss / batchsize * self.height_weight
        
#         return loss_dict

# class Joint_reg_Loss(nn.Module):
#     def __init__(self, device):
#         super(Joint_reg_Loss, self).__init__()
#         self.device = device
#         self.criterion_vert = nn.L1Loss().to(self.device)
#         self.criterion_joint = nn.MSELoss(reduction='none').to(self.device)
#         self.joint_weight = 5.0
#         self.verts_weight = 5.0
#         # self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

#     def forward(self, pred_joints, gt_joints, has_3d):
#         loss_dict = {}
        
#         # pred_joints = pred_joints[:,self.halpe2lsp]
#         # gt_joints = gt_joints[:,self.halpe2lsp]

#         conf = gt_joints[:, :, -1].unsqueeze(-1).clone()[has_3d == 1]

#         # gt_pelvis = (gt_joints[:,2,:3] + gt_joints[:,3,:3]) / 2.
#         gt_pelvis = gt_joints[:,19,:3]
#         gt_joints[:,:,:-1] = gt_joints[:,:,:-1] - gt_pelvis[:,None,:]

#         # pred_pelvis = (pred_joints[:,2,:] + pred_joints[:,3,:]) / 2.
#         pred_pelvis = pred_joints[:,19,:3]
#         pred_joints = pred_joints - pred_pelvis[:,None,:]

#         gt_joints = gt_joints[has_3d == 1]
#         pred_joints = pred_joints[has_3d == 1]

#         if len(gt_joints) > 0:
#             joint_loss = (conf * self.criterion_joint(pred_joints, gt_joints[:, :, :-1])).mean()
#         else:
#             joint_loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

#         loss_dict['Joint_reg_Loss'] = joint_loss * self.joint_weight
#         return loss_dict

# class Shape_reg(nn.Module):
#     def __init__(self, device):
#         super(Shape_reg, self).__init__()
#         self.device = device
#         self.reg_weight = 0.001

#     def forward(self, pred_shape):
#         loss_dict = {}
        
#         loss = torch.norm(pred_shape, dim=1)
#         loss = loss.mean()


#         loss_dict['shape_reg_loss'] = loss * self.reg_weight
#         return loss_dict

# def load_vposer():
#     import torch
#     # from  model.VPoser import VPoser

#     # settings of Vposer++
#     num_neurons = 512
#     latentD = 32
#     data_shape = [1,23,3]
#     trained_model_fname = 'data/vposer_snapshot.pkl' #'data/TR00_E096.pt'
    
#     vposer_pt = VPoser(num_neurons=num_neurons, latentD=latentD, data_shape=data_shape)

#     model_dict = vposer_pt.state_dict()
#     premodel_dict = torch.load(trained_model_fname)
#     premodel_dict = {k: v for k ,v in premodel_dict.items() if k in model_dict}
#     model_dict.update(premodel_dict)
#     vposer_pt.load_state_dict(model_dict)
#     print("load pretrain parameters from %s" %trained_model_fname)

#     vposer_pt.eval()

#     return vposer_pt

# class Pose_reg(nn.Module):
#     def __init__(self, device):
#         super(Pose_reg, self).__init__()
#         self.device = device
#         self.prior = load_vposer()
#         self.prior.to(self.device)

#         self.reg_weight = 0.001

#     def forward(self, pred_pose):
#         loss_dict = {}

#         z_mean = self.prior.encode_mean(pred_pose[:,3:])
#         loss = torch.norm(z_mean, dim=1)
#         loss = loss.mean()

#         loss_dict['pose_reg_loss'] = loss * self.reg_weight
#         return loss_dict

# class L2(nn.Module):
#     def __init__(self, device):
#         super(L2, self).__init__()
#         self.device = device
#         self.L2Loss = nn.MSELoss(reduction='sum')

#     def forward(self, x, y):
#         b = x.shape[0]
#         diff = self.L2Loss(x, y)
#         diff = diff / b
#         return diff

# class Smooth6D(nn.Module):
#     def __init__(self, device):
#         super(Smooth6D, self).__init__()
#         self.device = device
#         self.L1Loss = nn.L1Loss(size_average=False)

#     def forward(self, x, y):
#         b, f = x.shape[:2]
#         diff = self.L1Loss(x, y)
#         diff = diff / b / f
#         return diff

class MPJPE(nn.Module):
    def __init__(self, device):
        super(MPJPE, self).__init__()
        self.device = device
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

    def forward_instance(self, pred_joints, gt_joints, valid):
        loss_dict = {}

        # pred_joints = pred_joints[valid == 1]
        # gt_joints = gt_joints[valid == 1]

        conf = gt_joints[:,self.halpe2lsp,-1]

        pred_joints = pred_joints[:,self.halpe2lsp]
        gt_joints = gt_joints[:,self.halpe2lsp,:3]

        pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

        diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=[2]) * conf)
        diff = torch.mean(diff, dim=[1])
        diff = diff * 1000
        
        return diff
    
    def forward_instance_wrt_one(self, pred_joints, pred_trans, gt_joints, gt_trans, dshape, valid):
        '''
        将两个人的关节位置对齐到第一个人的pelvis位置，计算mpjpe
        '''
        # pred_joints = pred_joints[valid == 1]
        # gt_joints = gt_joints[valid == 1]
        # pred_trans = pred_trans[valid == 1]
        # gt_trans = gt_trans[valid == 1]
        # joints in camera coordinate
        pred_joints = pred_joints + pred_trans[:,None,:]
        gt_joints = gt_joints[...,:3] + gt_trans[:,None,:]
        
        pred_joints = pred_joints.reshape(dshape[0], dshape[1], dshape[2], -1, 3)[...,self.halpe2lsp,:] # (bs, frame_length, num_people, 26, 3)
        gt_joints = gt_joints.reshape(dshape[0], dshape[1], dshape[2], -1, 3)[...,self.halpe2lsp,:]
        valid = valid.reshape(dshape[0], dshape[1], dshape[2],)[:,:,0]
        
        # Align by pelvis of the first person
        pred_joints_a = pred_joints[:,:,0] # (bs, frame_length, 26, 3)
        gt_joints_a = gt_joints[:,:,0]
        
        pred_pelvis_a = (pred_joints_a[:,:,2] + pred_joints_a[:,:,3]) / 2.
        gt_pelvis_a = (gt_joints_a[:,:,2] + gt_joints_a[:,:,3]) / 2.
        pred_joints_wa = pred_joints - pred_pelvis_a[:,:,None,None,:]
        gt_joints_wa = gt_joints - gt_pelvis_a[:,:,None,None,:]
                
        diff = torch.sqrt(torch.sum((pred_joints_wa - gt_joints_wa)**2, dim=-1))
        diff_instance = diff.mean(dim=[-1])
        diff = diff.mean(dim=[-1,-2])
        
        
        # Align by pelvis of the second person
        pred_joints_b = pred_joints[:,:,1]
        gt_joints_b = gt_joints[:,:,1]
        
        pred_pelvis_b = (pred_joints_b[:,:,2] + pred_joints_b[:,:,3]) / 2.
        gt_pelvis_b = (gt_joints_b[:,:,2] + gt_joints_b[:,:,3]) / 2.
        pred_joints_wb = pred_joints - pred_pelvis_b[:,:,None,None,:]
        gt_joints_wb = gt_joints - gt_pelvis_b[:,:,None,None,:] # (bs, frame_length, num_people, 26, 3)
        
        diff_b = torch.sqrt(torch.sum((pred_joints_wb - gt_joints_wb)**2, dim=-1)) # (bs, frame_length, num_people, 26)
        diff_instance_b = diff_b.mean(dim=[-1]) # (bs, frame_length, num_people)
        diff_b = diff_b.mean(dim=[-1,-2]) # (bs, frame_length)
        
        dmask = diff < diff_b
        diff_instance = diff_instance * dmask[:,:,None] + diff_instance_b * (~dmask)[:,:,None]
                
        return diff_instance * 1000
    
    def forward(self, pred_joints, pred_trans, gt_joints, gt_trans, valid, dshape):
        loss_dict = {}
        
        gmpjpe = self.forward_gmpjpe(pred_joints, pred_trans, gt_joints, gt_trans, dshape, valid)
        loss_dict['G-MPJPE'] = gmpjpe

        mpjpe = self.forward_mpjpe(pred_joints, gt_joints, valid)
        loss_dict['MPJPE'] = mpjpe
        
        pampjpe = self.forwar_pampjpe(pred_joints, gt_joints, valid)
        loss_dict['PA-MPJPE'] = pampjpe
        
        mpjpe_wrt_one = self.forward_mpjpr_wrt_one(pred_joints, pred_trans, gt_joints, gt_trans, dshape, valid)
        loss_dict['MPJPE_wrt_one'] = mpjpe_wrt_one
              
        
        return loss_dict
    
    def forward_gmpjpe(self, pred_joints, pred_trans, gt_joints, gt_trans, dshape, valid):
        '''
        将两个人的关节位置对齐到第一个人的pelvis位置，计算mpjpe
        '''
        if valid.sum() == 0:
            return torch.FloatTensor(1).fill_(0.).to(self.device)[0]
                
        # joints in camera coordinate
        pred_joints = pred_joints + pred_trans[:,None,:]
        gt_joints = gt_joints[...,:3] + gt_trans[:,None,:]
        
        pred_joints = pred_joints[:,self.halpe2lsp]
        gt_joints = gt_joints[:,self.halpe2lsp,:3]
                
        diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=-1)) # [bs*frame_length*agent_num, lsp_j_num]
        diff = diff.mean(dim=-1) * valid
        diff = diff.sum()
        diff = diff * 1000
                
        return diff
    
    def forward_mpjpe(self, pred_joints, gt_joints, valid):      

        if valid.sum() == 0:
            return torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        conf = gt_joints[:,self.halpe2lsp,-1]

        pred_joints = pred_joints[:,self.halpe2lsp]
        gt_joints = gt_joints[:,self.halpe2lsp,:3]

        pred_pelvis = (pred_joints[...,2,:] + pred_joints[...,3,:]) / 2.
        gt_pelvis = (gt_joints[...,2,:] + gt_joints[...,3,:]) / 2.
        
        pred_joints = pred_joints - pred_pelvis[:,None,:]
        gt_joints = gt_joints - gt_pelvis[:,None,:]

        diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=-1) * conf) # [bs*frame_length*agent_num, lsp_j_num]
        diff = diff.mean(dim=-1) * valid
        diff = diff.sum()
        diff = diff * 1000
        
        return diff
    
    def forward_mpjpr_wrt_one(self, pred_joints, pred_trans, gt_joints, gt_trans, dshape, valid):
        '''
        将两个人的关节位置对齐到第一个人的pelvis位置，计算mpjpe
        '''
        if valid.sum() == 0:
            return torch.FloatTensor(1).fill_(0.).to(self.device)[0]
        
        # pred_joints = pred_joints[valid == 1]
        # gt_joints = gt_joints[valid == 1]
        # pred_trans = pred_trans[valid == 1]
        # gt_trans = gt_trans[valid == 1]
        # joints in camera coordinate
        pred_joints = pred_joints + pred_trans[:,None,:]
        gt_joints = gt_joints[...,:3] + gt_trans[:,None,:]
        
        pred_joints = pred_joints.reshape(dshape[0], dshape[1], dshape[2], -1, 3)[...,self.halpe2lsp,:] # (bs, frame_length, num_people, 26, 3)
        gt_joints = gt_joints.reshape(dshape[0], dshape[1], dshape[2], -1, 3)[...,self.halpe2lsp,:]
        valid = valid.reshape(dshape[0], dshape[1], dshape[2],)[:,:,0]
        
        # Align by pelvis of the first person
        pred_joints_a = pred_joints[:,:,0] # (bs, frame_length, 26, 3)
        gt_joints_a = gt_joints[:,:,0]
        
        pred_pelvis_a = (pred_joints_a[:,:,2] + pred_joints_a[:,:,3]) / 2.
        gt_pelvis_a = (gt_joints_a[:,:,2] + gt_joints_a[:,:,3]) / 2.
        pred_joints_wa = pred_joints - pred_pelvis_a[:,:,None,None,:]
        gt_joints_wa = gt_joints - gt_pelvis_a[:,:,None,None,:]
                
        diff = torch.sqrt(torch.sum((pred_joints_wa - gt_joints_wa)**2, dim=-1))
        diff = diff.mean(dim=[-1,-2])
        
        # Align by pelvis of the second person
        pred_joints_b = pred_joints[:,:,1]
        gt_joints_b = gt_joints[:,:,1]
        
        pred_pelvis_b = (pred_joints_b[:,:,2] + pred_joints_b[:,:,3]) / 2.
        gt_pelvis_b = (gt_joints_b[:,:,3] + gt_joints_b[:,:,3]) / 2.
        pred_joints_wb = pred_joints - pred_pelvis_b[:,:,None,None,:]
        gt_joints_wb = gt_joints - gt_pelvis_b[:,:,None,None,:] # (bs, frame_length, num_people, 26, 3)
        
        diff_b = torch.sqrt(torch.sum((pred_joints_wb - gt_joints_wb)**2, dim=-1)) # (bs, frame_length, num_people, 26)
        diff_b = diff_b.mean(dim=[-1,-2]) # (bs, frame_length)
        # diff = diff if diff < diff_b else diff_b
        diff = torch.where(diff < diff_b, diff, diff_b)
        diff = diff * valid
        diff = diff.sum()
        diff = diff * 1000 * 2 # mul 2 for two people
                
        return diff

    def forwar_pampjpe(self, pred_joints, gt_joints, valid):
        if valid.sum() == 0:
            return torch.FloatTensor(1).fill_(0.).to(self.device)[0]
        # pred_joints = pred_joints[valid == 1]
        # gt_joints = gt_joints[valid == 1]

        conf = gt_joints[:,self.halpe2lsp,-1]

        pred_joints = pred_joints[:,self.halpe2lsp]
        gt_joints = gt_joints[:,self.halpe2lsp,:3]

        pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
        gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

        pred_joints = self.batch_compute_similarity_transform(pred_joints, gt_joints)

        diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=-1) * conf)
        diff = diff.mean(dim=-1) * valid
        diff = diff.sum()
        diff = diff * 1000
        
        return diff

    def batch_compute_similarity_transform(self, S1, S2):
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        '''
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.permute(0,2,1)
            S2 = S2.permute(0,2,1)
            transposed = True
        assert(S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=-1, keepdims=True)
        mu2 = S2.mean(axis=-1, keepdims=True)

        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = torch.sum(X1**2, dim=1).sum(dim=1)

        # 3. The outer product of X1 and X2.
        K = X1.bmm(X2.permute(0,2,1))

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, V = torch.svd(K)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
        Z = Z.repeat(U.shape[0],1,1)
        t1 = U.bmm(V.permute(0,2,1))
        t2 = torch.det(t1)
        Z[:,-1, -1] = Z[:,-1, -1] * torch.sign(t2)
        # Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

        # Construct R.
        R = V.bmm(Z.bmm(U.permute(0,2,1)))

        # 5. Recover scale.
        scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

        # 6. Recover translation.
        t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

        # 7. Error:
        S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

        if transposed:
            S1_hat = S1_hat.permute(0,2,1)

        return S1_hat

    def align_by_pelvis(self, joints, format='lsp'):
        """
        Assumes joints is 14 x 3 in LSP order.
        Then hips are: [3, 2]
        Takes mid point of these points, then subtracts it.
        """
        if format == 'lsp':
            left_id = 3
            right_id = 2
            pelvis = (joints[..., left_id, :] + joints[..., right_id, :]) / 2
        elif format in ['smpl', 'h36m']:
            pelvis_id = 0
            pelvis = joints[..., pelvis_id, :]
        elif format in ['mpi']:
            pelvis_id = 14
            pelvis = joints[..., pelvis_id, :]

        return joints - pelvis[:, None, :]

# class MPJPE_H36M(nn.Module):
#     def __init__(self, device):
#         super(MPJPE_H36M, self).__init__()
#         self.h36m_regressor = torch.from_numpy(np.load('data/smpl/J_regressor_h36m.npy')).to(torch.float32).to(device)
#         self.halpe_regressor = torch.from_numpy(np.load('data/smpl/J_regressor_halpe.npy')).to(torch.float32).to(device)
#         self.device = device
#         self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]
#         self.halpe2h36m = [19,12,14,16,11,13,15,19,19,18,17,5,7,9,6,8,10]
#         self.BEV_H36M_TO_J14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 0]

#     def forward_instance(self, pred_joints, gt_joints):
#         loss_dict = {}

#         conf = gt_joints[:,self.halpe2lsp,-1]

#         pred_joints = pred_joints[:,self.halpe2lsp]
#         gt_joints = gt_joints[:,self.halpe2lsp,:3]

#         pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
#         gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

#         diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=[2]) * conf)
#         diff = torch.mean(diff, dim=[1])
#         diff = diff * 1000
        
#         return diff.detach().cpu().numpy()

#     def forward(self, pred_joints, gt_joints):
#         loss_dict = {}

#         conf = gt_joints[:,:,-1]

#         h36m_joints = torch.matmul(self.h36m_regressor, pred_joints)
#         halpe_joints = torch.matmul(self.halpe_regressor, pred_joints)

#         pred_joints = halpe_joints[:,self.halpe2h36m]
#         pred_joints[:,[7,8,9,10]] = h36m_joints[:,[7,8,9,10]]
#         gt_joints = gt_joints[:,:,:3]

#         pred_joints = self.align_by_pelvis(pred_joints, format='h36m')
#         gt_joints = self.align_by_pelvis(gt_joints, format='h36m')

#         # gui.vis_skeleton(pred_joints.detach().cpu().numpy(), gt_joints.detach().cpu().numpy(), format='h36m')

#         # pred_joints = pred_joints[:,self.BEV_H36M_TO_J14]
#         # gt_joints = gt_joints[:,self.BEV_H36M_TO_J14]
#         # conf = conf[:,self.BEV_H36M_TO_J14]

#         diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=[2]) * conf)
#         diff = torch.mean(diff, dim=[1])
#         diff = torch.mean(diff) * 1000
        
#         return diff

#     def pa_mpjpe(self, pred_joints, gt_joints):
#         loss_dict = {}

#         conf = gt_joints[:,self.halpe2lsp,-1].detach().cpu()

#         pred_joints = pred_joints[:,self.halpe2lsp].detach().cpu()
#         gt_joints = gt_joints[:,self.halpe2lsp,:3].detach().cpu()

#         pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
#         gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

#         pred_joints = self.batch_compute_similarity_transform(pred_joints, gt_joints)

#         diff = torch.sqrt(torch.sum((pred_joints - gt_joints)**2, dim=[2]) * conf)
#         diff = torch.mean(diff, dim=[1])
#         diff = torch.mean(diff) * 1000
        
#         return diff

#     def batch_compute_similarity_transform(self, S1, S2):
#         '''
#         Computes a similarity transform (sR, t) that takes
#         a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
#         where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
#         i.e. solves the orthogonal Procrutes problem.
#         '''
#         transposed = False
#         if S1.shape[0] != 3 and S1.shape[0] != 2:
#             S1 = S1.permute(0,2,1)
#             S2 = S2.permute(0,2,1)
#             transposed = True
#         assert(S2.shape[1] == S1.shape[1])

#         # 1. Remove mean.
#         mu1 = S1.mean(axis=-1, keepdims=True)
#         mu2 = S2.mean(axis=-1, keepdims=True)

#         X1 = S1 - mu1
#         X2 = S2 - mu2

#         # 2. Compute variance of X1 used for scale.
#         var1 = torch.sum(X1**2, dim=1).sum(dim=1)

#         # 3. The outer product of X1 and X2.
#         K = X1.bmm(X2.permute(0,2,1))

#         # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
#         # singular vectors of K.
#         U, s, V = torch.svd(K)

#         # Construct Z that fixes the orientation of R to get det(R)=1.
#         Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
#         Z = Z.repeat(U.shape[0],1,1)
#         t1 = U.bmm(V.permute(0,2,1))
#         t2 = torch.det(t1)
#         Z[:,-1, -1] = Z[:,-1, -1] * torch.sign(t2)
#         # Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

#         # Construct R.
#         R = V.bmm(Z.bmm(U.permute(0,2,1)))

#         # 5. Recover scale.
#         scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

#         # 6. Recover translation.
#         t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

#         # 7. Error:
#         S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

#         if transposed:
#             S1_hat = S1_hat.permute(0,2,1)

#         return S1_hat

#     def align_by_pelvis(self, joints, format='lsp'):
#         """
#         Assumes joints is 14 x 3 in LSP order.
#         Then hips are: [3, 2]
#         Takes mid point of these points, then subtracts it.
#         """
#         if format == 'lsp':
#             left_id = 3
#             right_id = 2

#             pelvis = (joints[:,left_id, :] + joints[:,right_id, :]) / 2.
#         elif format in ['smpl', 'h36m']:
#             pelvis_id = 0
#             pelvis = joints[:,pelvis_id, :]
#         elif format in ['mpi']:
#             pelvis_id = 14
#             pelvis = joints[:,pelvis_id, :]

#         return joints - pelvis[:,None,:]

# class PCK(nn.Module):
#     def __init__(self, device):
#         super(PCK, self).__init__()
#         self.device = device
#         self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

#     def forward_instance(self, pred_joints, gt_joints):
#         loss_dict = {}
#         confs = gt_joints[:,self.halpe2lsp][:,:,-1]
#         pred_joints = pred_joints[:,self.halpe2lsp]
#         gt_joints = gt_joints[:,self.halpe2lsp][:,:,:3]

#         pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
#         gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

#         joint_error = torch.sqrt(torch.sum((pred_joints - gt_joints) ** 2, dim=-1) * confs)
#         diff = torch.mean((joint_error < 0.15).float(), dim=1)
#         diff = diff * 100
        
#         return diff.detach().cpu().numpy()

#     def forward(self, pred_joints, gt_joints):
#         loss_dict = {}
#         confs = gt_joints[:,self.halpe2lsp][:,:,-1].reshape(-1,)
#         pred_joints = pred_joints[:,self.halpe2lsp]
#         gt_joints = gt_joints[:,self.halpe2lsp][:,:,:3]

#         pred_joints = self.align_by_pelvis(pred_joints, format='lsp')
#         gt_joints = self.align_by_pelvis(gt_joints, format='lsp')

#         joint_error = torch.sqrt(torch.sum((pred_joints - gt_joints) ** 2, dim=-1)).reshape(-1,)
#         joint_error = joint_error[confs==1]
#         diff = torch.mean((joint_error < 0.15).float(), dim=0)
#         diff = diff * 100
        
#         return diff

#     def align_by_pelvis(self, joints, format='lsp'):
#         """
#         Assumes joints is 14 x 3 in LSP order.
#         Then hips are: [3, 2]
#         Takes mid point of these points, then subtracts it.
#         """
#         if format == 'lsp':
#             left_id = 3
#             right_id = 2

#             pelvis = (joints[:,left_id, :] + joints[:,right_id, :]) / 2.
#         elif format in ['smpl', 'h36m']:
#             pelvis_id = 0
#             pelvis = joints[pelvis_id, :]
#         elif format in ['mpi']:
#             pelvis_id = 14
#             pelvis = joints[pelvis_id, :]

#         return joints - pelvis[:,None,:].repeat(1, 14, 1)

from utils._smpl import Mesh
class ContactCls_Loss(nn.Module):
    def __init__(self, device):
        super(ContactCls_Loss, self).__init__()
        self.device = device
        self.cls_loss = nn.BCELoss()
        self.contact_weight = 1.
        self.mesh_sampler = Mesh()

    def forward(self, pred_afford_full, pred_afford_sub, pred_afford_sub2, contact_corres):
        loss_dict = {}

        gt_afford_full = torch.where(contact_corres > 0, 1., 0.).reshape(-1,6890)
        gt_afford_sub2 = self.mesh_sampler.downsample(gt_afford_full.reshape(-1,6890,1), n1=0, n2=2)
        gt_afford_sub = self.mesh_sampler.downsample(gt_afford_full.reshape(-1,6890,1))

        pred_afford_full = pred_afford_full.reshape(-1,6890)
        pred_afford_sub = pred_afford_sub.reshape_as(gt_afford_sub)
        pred_afford_sub2 = pred_afford_sub2.reshape_as(gt_afford_sub2)


        loss_dict['contact_cls_loss'] = self.cls_loss(pred_afford_full, gt_afford_full) + \
                                        self.cls_loss(pred_afford_sub, gt_afford_sub) + \
                                        self.cls_loss(pred_afford_sub2, gt_afford_sub2)
        
        
        return loss_dict
    
from utils.sdf_loss import SDFLoss

class SDF_Loss(nn.Module):
    def __init__(self, device, smpl, thresh=0.0, robustifier=None):
        super(SDF_Loss, self).__init__()
        self.device = device
        self.neutral_smpl = smpl
        self.sdf_weight = 1000.0
        self.sdf_loss = SDFLoss(self.neutral_smpl.faces, debugging=False, robustifier=robustifier).cuda()
        self.thresh = thresh
        self.eval_mode = False

    def forward(self, pred_verts, pred_cam_t, dshape, valid):
        loss_dict = {}
        
        if valid.sum() == 0:
            loss_dict['SDF_Loss'] = torch.tensor(0.0).to(self.device)
            return loss_dict
        
        bs, frame_length, num_people = dshape[0], dshape[1], dshape[2]
        
        pred_verts = pred_verts.reshape(bs, frame_length, num_people, -1, 3)
        pred_cam_t = pred_cam_t.reshape(bs, frame_length, num_people, 3)
        
        valid = valid.reshape(bs, frame_length, num_people)[:,:,0]
        
        loss = []
        iter = 0
        
        for batch in range(bs):
            for f in range(frame_length):
                sdf_loss = self.sdf_loss(pred_verts[batch, f], pred_cam_t[batch, f], iter = iter, threshold=self.thresh)
                loss.append(sdf_loss)
                iter += 1
        loss = torch.stack(loss, dim=0)
        loss = loss.reshape(bs, frame_length) * valid
        loss = loss.sum()
        if not self.eval_mode:
            loss = loss / valid.sum()
        loss_dict['SDF_Loss'] = loss*self.sdf_weight
        
        return loss_dict
    
    def set_eval_mode(self):
        self.sdf_weight = 1000.0
        self.eval_mode = True
        
class MPVPE(nn.Module):
    def __init__(self, device):
        super(MPVPE, self).__init__()
        self.device = device
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]
        self.verts_weight = 1000.0

    def forward(self, pred_vertices, gt_vertices, pred_joints, gt_joints, has_smpl, valid):
        loss_dict = {}

        if valid.sum() == 0:
            loss_dict['MPVPE'] = torch.tensor(0.0).to(self.device)
            return loss_dict
        
        # pred_vertices = pred_vertices[valid == 1]
        # gt_vertices = gt_vertices[valid == 1]
        # pred_joints = pred_joints[valid == 1]
        # gt_joints = gt_joints[valid == 1]
        
        pred_joints = pred_joints[:,self.halpe2lsp]
        gt_joints = gt_joints[:,self.halpe2lsp,:3]
        
        premesh = self.align_mesh_by_pelvis_batch(pred_vertices, pred_joints, format='lsp')
        gt_mesh = self.align_mesh_by_pelvis_batch(gt_vertices, gt_joints, format='lsp')
        
        vertex_errors = torch.mean(torch.sqrt(torch.sum((gt_mesh - premesh) ** 2, dim=-1)), dim=-1) # [bs*frame_length*[agent_num]]
        vertex_errors = vertex_errors * valid
        vertex_errors = vertex_errors.sum()
        loss_dict['MPVPE'] = vertex_errors * self.verts_weight
        return loss_dict
    
    def align_mesh_by_pelvis_batch(self, mesh, joints, get_pelvis=False, format='lsp'):
        """
        Assumes joints is 14 x 3 in LSP order.
        Then hips are: [3, 2]
        Takes mid point of these points, then subtracts it.
        """
        if format == 'lsp':
            left_id = 3
            right_id = 2
            pelvis = (joints[:,left_id, :] + joints[:,right_id, :]) / 2.
        elif format in ['smpl', 'h36m']:
            pelvis_id = 0
            pelvis = joints[:,pelvis_id, :]
        elif format in ['mpi']:
            pelvis_id = 14
            pelvis = joints[:,pelvis_id, :]
        if get_pelvis:
            return mesh - pelvis[:,None], pelvis
        else:
            return mesh - pelvis[:,None]
    
from model.smpl2capsule import Smpl2Capscule, save_capsule_model, render_capsule_mesh
class Approx_Collision_Loss(nn.Module):
    def __init__(self, device, smpl):
        super(Approx_Collision_Loss, self).__init__()
        self.J_regressor = torch.tensor(smpl.J_regressor, dtype=torch.float32, device=device)                                 
        self.device = device
        self.smpl2capsule = Smpl2Capscule().to(device)
        state_dict = torch.load('data/smpl2cap.ckpt')['state_dict']
        # remove the prefix 'model.'
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[6:]] = v
        self.smpl2capsule.load_state_dict(new_state_dict)
        self.smpl2capsule.eval()
        self.capsule_num = self.smpl2capsule.total_capsule_num
        self.collision_weight = 1000000.0
        self.collision_thresh = 0.0
        
    def forward(self, pred_betas, gt_betas, pred_verts, gt_verts, pred_joints_halpe, gt_joints_halpe, pred_cam_t, gt_cam_t, dshape, valid):
        loss_dict = {}
        
        pred_joints = torch.tensordot(pred_verts, self.J_regressor.transpose(0, 1), dims=([1], [0])).transpose(1, 2)
        pred_joints_halpe = pred_joints_halpe[...,:3] + pred_cam_t[:,None,:]
        pred_joints = pred_joints + pred_cam_t[:,None,:]
        capsules_pred = self.smpl2capsule(pred_betas, pred_joints, pred_joints_halpe)
        
        capsules_pred = capsules_pred.reshape(dshape[0], dshape[1], dshape[2], self.capsule_num, 8)#[:,:,:,cap_idx]
                    
        # Get capsules for each person
        caps_pred_person1 = capsules_pred[:, :, 0]  # [bs, seq_len, num_capsules, 8]
        caps_pred_person2 = capsules_pred[:, :, 1]  # [bs, seq_len, num_capsules, 8]
        
        # Compute distances between all capsule pairs
        caps_dist_mtx = self.smpl2capsule.compute_capsule_distance(caps_pred_person1, caps_pred_person2)  # [bs, seq_len, c1, c2]
        
        repulsion_loss = torch.where(caps_dist_mtx < -self.collision_thresh, caps_dist_mtx ** 2, torch.zeros_like(caps_dist_mtx)) # [bs, seq_len, c1, c2]

        # Compute collision loss
        collision_loss = repulsion_loss.mean()
        
        if collision_loss != collision_loss or collision_loss > 10.0:
            collision_loss = torch.tensor(0.0).to(self.device)
        
        loss_dict['Approx_Collision_Loss'] = collision_loss * self.collision_weight
            
        return loss_dict

class Contact_Distance_Loss(nn.Module):
    def __init__(self, device):
        super(Contact_Distance_Loss, self).__init__()
        self.device = device
        self.contact_weight = 10.0
        # self.sparseSMPL = np.load('data/sparseSMPL.npy')

    def forward(self, pred_verts, pert_cam_t, dshape, gt_contact, valid):
        # TODO: extract valid data
        loss_dict = {}
        contact_loss = torch.tensor(0.0, device=self.device)
        
        pred_verts = pred_verts + pert_cam_t[:, None, :]
        pred_verts = pred_verts.reshape(dshape[0], dshape[1], dshape[2], 6890, 3)
        gt_contact = gt_contact.reshape(dshape[0], dshape[1], dshape[2], 6890).long()

        verts_a = pred_verts[:, :, 0, :, :]
        verts_b = pred_verts[:, :, 1, :, :] #  [bs, seq_len, num_verts, 3]
        gt_contact_a = gt_contact[:, :, 0, :] # [bs, seq_len, num_verts]
        
        # Downsample vertices
        # verts_a = verts_a[:, :, self.sparseSMPL]
        # verts_b = verts_b[:, :, self.sparseSMPL]
        # gt_contact_a = gt_contact_a[:, :, self.sparseSMPL]

        # Find contact points        
        for b in range(dshape[0]):
            batch_verts_a = verts_a[b]
            batch_verts_b = verts_b[b]
            batch_contact_idx = gt_contact_a[b] # index of which vertices of b are in contact with a
            
            if batch_contact_idx.sum() == 0:
                continue
            
            contact_verts_a = batch_verts_a[batch_contact_idx>0] # points that have contact with b
            # rearrange verts_b to match the contact points
            rearranged_verts_b = None
            for f in range(dshape[1]):
                if batch_contact_idx[f].sum() == 0:
                    continue
                rearranged_verts_bf= batch_verts_b[f][batch_contact_idx[f].nonzero().squeeze()].reshape(-1, 3)
                if rearranged_verts_b is None:
                    rearranged_verts_b = rearranged_verts_bf
                else:
                    rearranged_verts_b = torch.cat((rearranged_verts_b, rearranged_verts_bf), dim=0)
            rearranged_verts_b = rearranged_verts_b.reshape_as(contact_verts_a)
            
            # Compute distances
            batch_dist = torch.norm(contact_verts_a - rearranged_verts_b, dim=-1)
            contact_loss += batch_dist.mean()
            
            # Clear GPU memory
            del batch_verts_a, batch_verts_b, batch_dist
            torch.cuda.empty_cache()
            
        contact_loss = contact_loss / dshape[0]

        loss_dict['Contact_Distance_Loss'] = contact_loss * self.contact_weight

        return loss_dict
       
import os
import pickle
class GMM_Loss(nn.Module):
    def __init__(self, device, 
                prior_folder='./data',
                num_gaussians=8, dtype=np.float32, epsilon=1e-16,
                use_merged=True,
                 ):
        super(GMM_Loss, self).__init__()
        self.device = device
        self.gmm_weight = 0.01

        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged
        gmm_fn = 'gmm_{:02d}.pkl'.format(num_gaussians)
        
        full_gmm_fn = os.path.join(prior_folder, gmm_fn)
        try:
            with open(full_gmm_fn, 'rb') as f:
                gmm = pickle.load(f, encoding='latin1')
        except Exception as e:
            print('Error loading GMM from {}: {}'.format(full_gmm_fn, e))

        if type(gmm) == dict:
            means = gmm['means'].astype(dtype)
            covs = gmm['covars'].astype(dtype)
            weights = gmm['weights'].astype(dtype)
        elif 'sklearn.mixture.gmm.GMM' in str(type(gmm)):
            means = gmm.means_.astype(dtype)
            covs = gmm.covars_.astype(dtype)
            weights = gmm.weights_.astype(dtype)
        else:
            print('Unknown type for the prior: {}, exiting!'.format(type(gmm)))
            exit(-1)

        self.register_buffer('means', torch.tensor(means, dtype=torch.float))
        self.means = self.means.to(device)

        self.register_buffer('covs', torch.tensor(covs, dtype=torch.float32))
        self.covs = self.covs.to(device)

        precisions = [np.linalg.inv(cov) for cov in covs]
        precisions = np.stack(precisions).astype(dtype)

        self.register_buffer('precisions',
                             torch.tensor(precisions, dtype=torch.float32))
        self.precisions = self.precisions.to(device)

        # The constant term:
        sqrdets = np.array([(np.sqrt(np.linalg.det(c)))
                            for c in gmm['covars']])
        const = (2 * np.pi)**(69 / 2.)

        nll_weights = np.asarray(gmm['weights'] / (const *
                                                   (sqrdets / sqrdets.min())))
        nll_weights = torch.tensor(nll_weights, dtype=torch.float32).unsqueeze(dim=0)
        self.register_buffer('nll_weights', nll_weights)
        self.nll_weights = self.nll_weights.to(device)

        weights = torch.tensor(gmm['weights'], dtype=torch.float32).unsqueeze(dim=0)
        self.register_buffer('weights', weights)
        self.weights = self.weights.to(device)

        self.register_buffer('pi_term',
                             torch.log(torch.tensor(2 * np.pi, dtype=torch.float32)))
        self.pi_term = self.pi_term.to(device)

        cov_dets = [np.log(np.linalg.det(cov.astype(dtype)) + epsilon)
                    for cov in covs]
        self.register_buffer('cov_dets',
                             torch.tensor(cov_dets, dtype=torch.float32))
        self.cov_dets = self.cov_dets.to(device)

        # The dimensionality of the random variable
        self.random_var_dim = self.means.shape[1]

    def get_mean(self):
        ''' Returns the mean of the mixture '''
        mean_pose = torch.matmul(self.weights, self.means)
        return mean_pose

    def merged_log_likelihood(self, pose, betas):
        diff_from_mean = pose.unsqueeze(dim=1) - self.means

        prec_diff_prod = torch.einsum('mij,bmj->bmi',
                                      [self.precisions, diff_from_mean])
        diff_prec_quadratic = (prec_diff_prod * diff_from_mean).sum(dim=-1)

        curr_loglikelihood = 0.5 * diff_prec_quadratic - \
            torch.log(self.nll_weights)
        #  curr_loglikelihood = 0.5 * (self.cov_dets.unsqueeze(dim=0) +
        #  self.random_var_dim * self.pi_term +
        #  diff_prec_quadratic
        #  ) - torch.log(self.weights)

        min_likelihood, _ = torch.min(curr_loglikelihood, dim=1)
        return min_likelihood

    def log_likelihood(self, pose, betas, *args, **kwargs):
        ''' Create graph operation for negative log-likelihood calculation
        '''
        likelihoods = []

        for idx in range(self.num_gaussians):
            mean = self.means[idx]
            prec = self.precisions[idx]
            cov = self.covs[idx]
            diff_from_mean = pose - mean

            curr_loglikelihood = torch.einsum('bj,ji->bi',
                                              [diff_from_mean, prec])
            curr_loglikelihood = torch.einsum('bi,bi->b',
                                              [curr_loglikelihood,
                                               diff_from_mean])
            cov_term = torch.log(torch.det(cov) + self.epsilon)
            curr_loglikelihood += 0.5 * (cov_term +
                                         self.random_var_dim *
                                         self.pi_term)
            likelihoods.append(curr_loglikelihood)

        log_likelihoods = torch.stack(likelihoods, dim=1)
        min_idx = torch.argmin(log_likelihoods, dim=1)
        weight_component = self.nll_weights[:, min_idx]
        weight_component = -torch.log(weight_component)

        return weight_component + log_likelihoods[:, min_idx]

    def forward(self, pose, betas):
        loss_dict = {}
        pose = pose.reshape(-1, 72)[:,3:]
        betas = betas.reshape(-1, 10)
        if self.use_merged:
            loss = self.merged_log_likelihood(pose, betas)
        else:
            loss =  self.log_likelihood(pose, betas)
            
        loss = loss.mean()
        loss_dict['GMM_Loss'] = loss * self.gmm_weight
        return loss_dict


# class Global_Joint_Loss(nn.Module):
#     def __init__(self, device):
#         super(Global_Joint_Loss, self).__init__()
#         self.device = device
#         self.criterion_joint = nn.MSELoss(reduction='none').to(self.device)
#         self.mpjpe_weight = 1000.0
#         self.pa_mpjpe_weight = 1000.0
#         self.mpjpe_wrt_one_weight = 1000.0
#         # self.dpa_mpjpe_weight = 1.0

#     def forward(self, pred_joints, gt_joints, gt_trans, valid, dshape):
#         loss_dict = {}

#         gt_joints = gt_joints[..., :3] + gt_trans[..., None, :]

#         g_joint_loss = self.forward_g_joints(pred_joints, gt_joints, valid)
#         loss_dict['g_joint_loss'] = g_joint_loss * self.mpjpe_weight # in camera coordinate
        
#         ra_joints_loss = self.forward_ra_joints(pred_joints, gt_joints, valid)
#         loss_dict['ra_joints_loss'] = ra_joints_loss * self.pa_mpjpe_weight # aling by pelvis
        
#         joint_loss_wrt_one = self.forward_joints_wrt_one(pred_joints, gt_joints, dshape, valid)
#         loss_dict['joint_loss_wrt_one'] = joint_loss_wrt_one * self.mpjpe_wrt_one_weight # align by the first person's pelvis
              
#         # dpa_joints_loss = self.foward_dpa_joints(pred_joints, pred_trans, gt_joints, gt_trans, dshape, valid)
#         # loss_dict['dpa_joints_loss'] = dpa_joints_loss * self.dpa_mpjpe_weight # depth-aware joints loss
        
#         return loss_dict
    
#     def forward_g_joints(self, pred_joints, gt_joints, valid):
#         pred_joints = pred_joints.reshape_as(gt_joints[...,:3])
#         pred_joints = pred_joints[valid == 1]
#         gt_joints = gt_joints[valid == 1]
#         gt_joints = gt_joints[...,:3]

#         loss = self.criterion_joint(pred_joints, gt_joints).mean()
        
#         return loss
    
#     def forward_joints_wrt_one(self, pred_joints, gt_joints, dshape, valid):
#         pred_joints = pred_joints.reshape_as(gt_joints[...,:3])[valid == 1]
#         gt_joints = gt_joints[valid == 1]

#         pred_joints = pred_joints.reshape(dshape[0], dshape[1], dshape[2], -1, 3) # (bs, frame_length, num_people, 26, 3)
#         gt_joints = gt_joints.reshape(dshape[0], dshape[1], dshape[2], -1, 3)
        
#         pred_joints_a = pred_joints[:,:,0] # (bs, frame_length, 26, 3)
#         gt_joints_a = gt_joints[:,:,0]
        
#         # Align by pelvis of the first person
#         pred_pelvis_a = pred_joints_a[:,:,19]
#         gt_pelvis_a = gt_joints_a[:,:,19]
        
#         pred_joints = pred_joints - pred_pelvis_a[:,:,None,None,:]
#         gt_joints = gt_joints - gt_pelvis_a[:,:,None,None,:]
        
#         loss = self.criterion_joint(pred_joints, gt_joints).mean()
        
#         return loss

#     def forward_ra_joints(self, pred_joints, gt_joints, valid):
#         pred_joints = pred_joints.reshape_as(gt_joints[...,:3])
#         pred_joints = pred_joints[valid == 1]
#         gt_joints = gt_joints[valid == 1]

#         gt_pelvis = gt_joints[...,19,:3]
#         gt_joints = gt_joints[...,:3] - gt_pelvis[...,None,:]

#         pred_pelvis = pred_joints[...,19,:3]
#         pred_joints = pred_joints - pred_pelvis[:,None,:]

#         loss = self.criterion_joint(pred_joints, gt_joints).mean()
        
#         return loss
    
#     def foward_dpa_joints(self, pred_joints, pred_trans, gt_joints, gt_trans, dshape, valid):
#         """
#         depth-aware joints loss
#         """
#         # get the joints in camera coordinate
#         pred_joints = pred_joints + pred_trans[:,None,:]
#         gt_joints = gt_joints[...,:3] + gt_trans[:,None,:]
        
#         # reshape the joints
#         pred_joints = pred_joints.reshape(dshape[0]*dshape[1], dshape[2]*26, 3) # (bs*frame_length, num_people, 26, 3)
#         gt_joints = gt_joints.reshape(dshape[0]*dshape[1], dshape[2]*26, 3)
        
#         # get the depth order of joints
#         pred_order = self.calculate_depth_order(pred_joints)
#         gt_order = self.calculate_depth_order(gt_joints)
        
#         # calculate depth order loss
#         depth_order_loss = torch.log(1+torch.exp(pred_order - gt_order)).mean()
#         return depth_order_loss                
        
#     def calculate_depth_order(self, joints):
#         """
#         Calculate the depth order of joints based on the z-axis values.
#         Args:
#             joints: (bs, num_joints, 3) tensor representing the joints.
#         Returns:
#             depth_order: (bs, num_joints) tensor representing the depth order of joints.
#         """
#         depth_order = torch.argsort(joints[..., 2], dim=-1)
#         return depth_order

#     def batch_compute_similarity_transform(self, S1, S2):
#         '''
#         Computes a similarity transform (sR, t) that takes
#         a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
#         where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
#         i.e. solves the orthogonal Procrutes problem.
#         '''
#         transposed = False
#         if S1.shape[0] != 3 and S1.shape[0] != 2:
#             S1 = S1.permute(0,2,1)
#             S2 = S2.permute(0,2,1)
#             transposed = True
#         assert(S2.shape[1] == S1.shape[1])

#         # 1. Remove mean.
#         mu1 = S1.mean(axis=-1, keepdims=True)
#         mu2 = S2.mean(axis=-1, keepdims=True)

#         X1 = S1 - mu1
#         X2 = S2 - mu2

#         # 2. Compute variance of X1 used for scale.
#         var1 = torch.sum(X1**2, dim=1).sum(dim=1)

#         # 3. The outer product of X1 and X2.
#         K = X1.bmm(X2.permute(0,2,1))

#         # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
#         # singular vectors of K.
#         U, s, V = torch.svd(K)

#         # Construct Z that fixes the orientation of R to get det(R)=1.
#         Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
#         Z = Z.repeat(U.shape[0],1,1)
#         t1 = U.bmm(V.permute(0,2,1))
#         t2 = torch.det(t1)
#         Z[:,-1, -1] = Z[:,-1, -1] * torch.sign(t2)
#         # Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

#         # Construct R.
#         R = V.bmm(Z.bmm(U.permute(0,2,1)))

#         # 5. Recover scale.
#         scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

#         # 6. Recover translation.
#         t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

#         # 7. Error:
#         S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

#         if transposed:
#             S1_hat = S1_hat.permute(0,2,1)

#         return S1_hat

# class Global_Int_Loss(nn.Module):
#     def __init__(self, device):
#         super(Global_Int_Loss, self).__init__()
#         self.device = device
#         self.criterion_dm = nn.MSELoss().to(self.device)
#         self.interaction_weight = 10.0
#         self.num_people = 2

#     def forward(self, pred_joints, gt_joints, gt_trans, has_3d, valid):
#         loss_dict = {}
#         pred_joints = pred_joints.reshape_as(gt_joints[...,:3])[valid == 1]
#         gt_joints = gt_joints[...,:3] + gt_trans[...,None,:]
#         gt_joints = gt_joints[valid == 1]
        
#         pred_joints = pred_joints.reshape(-1, self.num_people, 26, 3) # [bs*frame_length, num_people, 26, 3]
#         gt_joints = gt_joints.reshape(-1, self.num_people, 26, 3)
        
#         pred_joints_a = pred_joints[:,0]
#         pred_joints_b = pred_joints[:,1]
#         gt_joints_a = gt_joints[:,0]
#         gt_joints_b = gt_joints[:,1]
        
#         # define distance map as every a joint to every b joint, an 26x26 matrix
#         pred_joints_distance_matrix = torch.cdist(pred_joints_a, pred_joints_b, p=2)
#         gt_joints_distance_matrix = torch.cdist(gt_joints_a, gt_joints_b, p=2)
        
#         # calculate the loss
#         # apply this loss only when the element in the distance map is less than 1
#         mask = (gt_joints_distance_matrix < 1).float()
#         # print(mask.sum())
#         int_loss = self.criterion_dm(pred_joints_distance_matrix, gt_joints_distance_matrix) * mask
#         int_loss = int_loss.mean()

#         loss_dict['Int_Loss'] = int_loss * self.interaction_weight

#         return loss_dict
    
#     def eval_mode(self):
#         self.interaction_weight = 1000.0

# class Global_Vel_Loss(nn.Module):
#     def __init__(self, device):
#         super(Global_Vel_Loss, self).__init__()
#         self.device = device
#         self.criterion_joint_vel = nn.MSELoss(reduction='none').to(self.device)
#         self.vel_weight = 5000.0
        
#     def forward(self, pred_joints, gt_joints, gt_transl, dshape, valid):
#         loss_dict = {}
#         pred_joints = pred_joints.reshape_as(gt_joints[...,:3])[valid == 1]
#         gt_joints = gt_joints[...,:3] + gt_transl[:,None,:]
#         gt_joints = gt_joints[valid == 1]

#         pred_joints = pred_joints.reshape(dshape[0], dshape[1], dshape[2], -1, 3)
#         gt_joints = gt_joints.reshape(dshape[0], dshape[1], dshape[2], -1, 3)

#         pred_vel = pred_joints[:,1:] - pred_joints[:,:1]
#         gt_vel = gt_joints[:,1:] - gt_joints[:,:1]

#         # pred_vel = pred_vel.reshape(dshape[0]*(dshape[1]-1)*dshape[2], -1, 3)
#         # gt_vel = gt_vel.reshape(dshape[0]*(dshape[1]-1)*dshape[2], -1, 3)
        
#         if len(gt_joints) > 0:
#             vel_loss = (self.criterion_joint_vel(pred_vel, gt_vel)).mean()
#         else:
#             vel_loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

#         loss_dict['Vel_Loss'] = vel_loss * self.vel_weight
#         return loss_dict
    
#     def forward_instance(self, pred_joints, pred_transl, gt_joints, gt_transl, dshape, has_3d, valid):
                
#         pred_joints = pred_joints + pred_transl[:,None,:]
#         gt_joints = gt_joints[...,:3] + gt_transl[:,None,:]

#         pred_joints = pred_joints.reshape(dshape[0], dshape[1], dshape[2], -1, 3)
#         gt_joints = gt_joints.reshape(dshape[0], dshape[1], dshape[2], -1, 3)

#         pred_vel = pred_joints[:,1:] - pred_joints[:,:1]
#         gt_vel = gt_joints[:,1:] - gt_joints[:,:1]

#         vel_loss_frame = self.criterion_joint_vel(pred_vel, gt_vel).mean(dim=[-1,-2,-3,-5])
        
#         return vel_loss_frame
import modules
class Psudo_MultiView_Keyp_Loss(nn.Module):
    def __init__(self, device):
        super(Psudo_MultiView_Keyp_Loss, self).__init__()
        self.device = device
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.keyp_weight = 1000.0
        self.halpe2lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,17]

    def forward(self, pred_joints, pred_trans, gt_joints, gt_trans, valid, img_h, img_w, focal_length):
        loss_dict = {}
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        
        pred_joints = pred_joints + pred_trans[:,None,:]
        gt_joints = gt_joints[...,:3] + gt_trans[:,None,:]

        # pred_joints = pred_joints[valid == 1]
        # gt_joints = gt_joints[valid == 1]
        gt_joints = gt_joints[...,:3]
        
        centroid1 = torch.mean(gt_joints, dim=1, keepdim=True) # (bs, 1, 3)
        centroid1[:, :, 2] = 0
        # rotate around y-axis for 90 degree
        aroundy = cv2.Rodrigues(np.array([np.radians(90.), 0, 0]))[0][np.newaxis, ...]  # 1*3*3
        aroundy = torch.FloatTensor(aroundy).to(self.device)
        # right side view
        pred_joints_new_view1 = self.batch_compute_similarity_transform(pred_joints - centroid1, aroundy) + centroid1
        gt_joints_new_view1 = self.batch_compute_similarity_transform(gt_joints - centroid1, aroundy) + centroid1
        # left side view
        pred_joints_new_view2 = self.batch_compute_similarity_transform(pred_joints_new_view1 - centroid1, -aroundy) + centroid1
        gt_joints_new_view2 = self.batch_compute_similarity_transform(gt_joints_new_view1 - centroid1, -aroundy) + centroid1
        # back side view
        aroundy2 = cv2.Rodrigues(np.array([np.radians(180.), 0, 0]))[0][np.newaxis, ...]  # 1*3*3
        aroundy2 = torch.FloatTensor(aroundy2).to(self.device)
        pred_joints_new_view3 = self.batch_compute_similarity_transform(pred_joints - centroid1, aroundy2) + centroid1
        gt_joints_new_view3 = self.batch_compute_similarity_transform(gt_joints - centroid1, aroundy2) + centroid1
        
        centroid2 = torch.mean(gt_joints_new_view1, dim=1, keepdim=True) # (bs, 1, 3)
        centroid2[:, :, 2] = 0
        aroundx = cv2.Rodrigues(np.array([np.radians(-90.), 0, 0]))[0][np.newaxis, ...]
        aroundx = torch.FloatTensor(aroundx).to(self.device)
        # top view
        pred_joints_new_view4 = self.batch_compute_similarity_transform(pred_joints_new_view1 - centroid2, aroundx) + centroid2
        gt_joints_new_view4 = self.batch_compute_similarity_transform(gt_joints_new_view1 - centroid2, aroundx) + centroid2
        # bottom view
        pred_joints_new_view5 = self.batch_compute_similarity_transform(pred_joints_new_view3 - centroid2, -aroundx) + centroid2
        gt_joints_new_view5 = self.batch_compute_similarity_transform(gt_joints_new_view3 - centroid2, -aroundx) + centroid2
        
        
        num_valid = pred_joints.shape[0]
        camera_center = torch.stack([img_w/2, img_h/2], dim=-1)
        camera_center = torch.FloatTensor([0, 0]).to(self.device)
        
        pred_kpts_new_view1 = perspective_projection(pred_joints_new_view1, 
                                                rotation=torch.eye(3, device=pred_joints.device).unsqueeze(0).expand(num_valid, -1, -1),
                                                translation=torch.zeros(3, device=pred_joints.device).unsqueeze(0).expand(num_valid, -1),
                                                focal_length=focal_length,
                                                camera_center=camera_center)
        gt_kpts_new_view1 = perspective_projection(gt_joints_new_view1,
                                                rotation=torch.eye(3, device=gt_joints.device).unsqueeze(0).expand(num_valid, -1, -1),
                                                translation=torch.zeros(3, device=gt_joints.device).unsqueeze(0).expand(num_valid, -1),
                                                focal_length=focal_length,
                                                camera_center=camera_center)
        pred_kpts_new_view2 = perspective_projection(pred_joints_new_view2,
                                                rotation=torch.eye(3, device=pred_joints.device).unsqueeze(0).expand(num_valid, -1, -1),
                                                translation=torch.zeros(3, device=pred_joints.device).unsqueeze(0).expand(num_valid, -1),
                                                focal_length=focal_length,
                                                camera_center=camera_center)
        gt_kpts_new_view2 = perspective_projection(gt_joints_new_view2,
                                                rotation=torch.eye(3, device=gt_joints.device).unsqueeze(0).expand(num_valid, -1, -1),
                                                translation=torch.zeros(3, device=gt_joints.device).unsqueeze(0).expand(num_valid, -1),
                                                focal_length=focal_length,
                                                camera_center=camera_center)
        pred_kpts_new_view3 = perspective_projection(pred_joints_new_view3,
                                                rotation=torch.eye(3, device=pred_joints.device).unsqueeze(0).expand(num_valid, -1, -1),
                                                translation=torch.zeros(3, device=pred_joints.device).unsqueeze(0).expand(num_valid, -1),
                                                focal_length=focal_length,
                                                camera_center=camera_center)
        gt_kpts_new_view3 = perspective_projection(gt_joints_new_view3,
                                                rotation=torch.eye(3, device=gt_joints.device).unsqueeze(0).expand(num_valid, -1, -1),
                                                translation=torch.zeros(3, device=gt_joints.device).unsqueeze(0).expand(num_valid, -1),
                                                focal_length=focal_length,
                                                camera_center=camera_center)
        pred_kpts_new_view4 = perspective_projection(pred_joints_new_view4,
                                                rotation=torch.eye(3, device=pred_joints.device).unsqueeze(0).expand(num_valid, -1, -1),
                                                translation=torch.zeros(3, device=pred_joints.device).unsqueeze(0).expand(num_valid, -1),
                                                focal_length=focal_length,
                                                camera_center=camera_center)
        gt_kpts_new_view4 = perspective_projection(gt_joints_new_view4,
                                                rotation=torch.eye(3, device=gt_joints.device).unsqueeze(0).expand(num_valid, -1, -1),
                                                translation=torch.zeros(3, device=gt_joints.device).unsqueeze(0).expand(num_valid, -1),
                                                focal_length=focal_length,
                                                camera_center=camera_center)
        pred_kpts_new_view5 = perspective_projection(pred_joints_new_view5,
                                                rotation=torch.eye(3, device=pred_joints.device).unsqueeze(0).expand(num_valid, -1, -1),
                                                translation=torch.zeros(3, device=pred_joints.device).unsqueeze(0).expand(num_valid, -1),
                                                focal_length=focal_length,
                                                camera_center=camera_center)
        gt_kpts_new_view5 = perspective_projection(gt_joints_new_view5,
                                                rotation=torch.eye(3, device=gt_joints.device).unsqueeze(0).expand(num_valid, -1, -1),
                                                translation=torch.zeros(3, device=gt_joints.device).unsqueeze(0).expand(num_valid, -1),
                                                focal_length=focal_length,
                                                camera_center=camera_center)
        
        results = {}
        results.update(pred_keypoints_2d = pred_kpts_new_view1.detach().cpu().numpy().astype(np.float32))
        results.update(psudu_gt_keypoints_2d = gt_kpts_new_view1.detach().cpu().numpy().astype(np.float32))
        results.update(gt_keypoints_2d = gt_kpts_new_view1.detach().cpu().numpy().astype(np.float32))
        modules.ModelLoader.save_2djoints(results, 0, 128, True)

        
        loss = 0
        loss += (self.criterion_keypoints(pred_kpts_new_view1, gt_kpts_new_view1)).mean()
        loss += (self.criterion_keypoints(pred_kpts_new_view2, gt_kpts_new_view2)).mean()
        loss += (self.criterion_keypoints(pred_kpts_new_view3, gt_kpts_new_view3)).mean()
        loss += (self.criterion_keypoints(pred_kpts_new_view4, gt_kpts_new_view4)).mean()
        loss += (self.criterion_keypoints(pred_kpts_new_view5, gt_kpts_new_view5)).mean()
        
        if loss > 300*5 or loss != loss:
            loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]

        loss_dict['Psudo_MultiView_Keyp_Loss'] = loss * self.keyp_weight
        return loss_dict
    
from utils.quaternion import qbetween
class Relative_Rotation_Loss(nn.Module):
    def __init__(self, device):
        super(Relative_Rotation_Loss, self).__init__()
        self.device = device
        self.rotation_weight = 10.0
        self.criterion_rot = nn.MSELoss(reduction='none').to(self.device)
        self.r_hip = 12
        self.l_hip = 11
        self.r_sdr = 6
        self.l_sdr = 5
        
    def forward(self, pred_joints, pred_trans, gt_joints, gt_trans, valid, dshape):

        # joints in camera coordinate
        pred_joints = pred_joints + pred_trans[:,None,:]
        gt_joints = gt_joints[...,:3] + gt_trans[:,None,:]

        # pred_joints = pred_joints[valid == 1]
        # gt_joints = gt_joints[valid == 1]
        
        pred_joints = pred_joints.reshape(dshape[0], dshape[1], dshape[2], -1, 3) # (bs, frame_length, num_people, 26, 3)
        gt_joints = gt_joints.reshape(dshape[0], dshape[1], dshape[2], -1, 3)
        
        across = pred_joints[..., self.r_hip, :] - pred_joints[..., self.l_hip, :]
        across = across / torch.norm(across, dim=-1, keepdim=True)
        across_gt = gt_joints[..., self.r_hip, :] - gt_joints[..., self.l_hip, :]
        across_gt = across_gt / torch.norm(across_gt, dim=-1, keepdim=True)
        
        y_axis = torch.zeros_like(across).to(self.device)
        y_axis[..., 1] = 1
        
        forward = torch.cross(across, y_axis, dim=-1)
        forward = forward / torch.norm(forward, dim=-1, keepdim=True)
        forward_gt = torch.cross(across_gt, y_axis, dim=-1)
        forward_gt = forward_gt / torch.norm(forward_gt, dim=-1, keepdim=True)
        
        pred_relative_rot = qbetween(forward[..., 0, :], forward[..., 1, :])
        tgt_relative_rot = qbetween(forward_gt[..., 0, :], forward_gt[..., 1, :])
        
        loss = self.criterion_rot(pred_relative_rot, tgt_relative_rot)
        loss = loss.mean()
        if loss > 300 or loss != loss:
            loss = torch.FloatTensor(1).fill_(0.).to(self.device)[0]
        loss_dict = {}
        loss_dict['Relative_Rotation_Loss'] = loss * self.rotation_weight
        return loss_dict
    
class Foot_Slipping_Loss(nn.Module):
    def __init__(self, device):
        super(Foot_Slipping_Loss, self).__init__()
        self.device = device
        self.foot_slipping_weight = 50.0
        self.foot_slipping_loss = nn.MSELoss(reduction='none').to(self.device)
        self.lfoot_ids = [15, 22]
        self.rfoot_ids = [16, 23]
        self.eval_mode = False
        
    def forward(self, pred_joints, pred_trans, dshape, valid):
        loss_dict = {}
        
        pred_joints = pred_joints + pred_trans[...,None,:]
        pred_joints = pred_joints.reshape(dshape[0], dshape[1], dshape[2], -1, 3)
        valid = valid.reshape(dshape[0], dshape[1], dshape[2], -1, 1)
        
        pred_joints_a = pred_joints[:,:,0] # [bs, seq_len, 24, 3]
        pred_joints_b = pred_joints[:,:,1]

        lfoot_vel_a = pred_joints_a[..., 1:, self.lfoot_ids,:] - pred_joints_a[..., :-1, self.lfoot_ids,:] # [bs, seq_len-1, 2, 3]
        lfoot_vel_a = lfoot_vel_a.mean(dim=-2) # [bs, seq_len-1, 3]
        rfoot_vel_a = pred_joints_a[..., 1:, self.rfoot_ids,:] - pred_joints_a[..., :-1, self.rfoot_ids,:]
        rfoot_vel_a = rfoot_vel_a.mean(dim=-2)

        lfoot_h_a = pred_joints_a[..., :-1, self.lfoot_ids, 1].mean(dim=-1) # [bs, seq_len-1]
        rfoot_h_a = pred_joints_a[..., :-1, self.rfoot_ids, 1].mean(dim=-1)

        lfoot_slip_a = torch.where(lfoot_h_a < rfoot_h_a, lfoot_vel_a.norm(dim=-1), torch.zeros_like(lfoot_vel_a.norm(dim=-1))) # [bs, seq_len]
        rfoot_slip_a = torch.where(rfoot_h_a < lfoot_h_a, rfoot_vel_a.norm(dim=-1), torch.zeros_like(rfoot_vel_a.norm(dim=-1))) # [bs, seq_len]

        lfoot_vel_b = pred_joints_b[..., 1:, self.lfoot_ids,:] - pred_joints_b[..., :-1, self.lfoot_ids,:]
        lfoot_vel_b = lfoot_vel_b.mean(dim=-2)
        rfoot_vel_b = pred_joints_b[..., 1:, self.rfoot_ids,:] - pred_joints_b[..., :-1, self.rfoot_ids,:]
        rfoot_vel_b = rfoot_vel_b.mean(dim=-2)

        lfoot_h_b = pred_joints_b[..., :-1, self.lfoot_ids, 1].mean(dim=-1)
        rfoot_h_b = pred_joints_b[..., :-1, self.rfoot_ids, 1].mean(dim=-1)

        lfoot_slip_b = torch.where(lfoot_h_b < rfoot_h_b, lfoot_vel_b.norm(dim=-1), torch.zeros_like(lfoot_vel_b.norm(dim=-1))) # [bs, seq_len]
        rfoot_slip_b = torch.where(rfoot_h_b < lfoot_h_b, rfoot_vel_b.norm(dim=-1), torch.zeros_like(rfoot_vel_b.norm(dim=-1))) # [bs, seq_len]
        
        foot_slipping_loss = lfoot_slip_a + rfoot_slip_a + lfoot_slip_b + rfoot_slip_b
        foot_slipping_loss = foot_slipping_loss * valid[:,:,0]
        if self.eval_mode:
            foot_slipping_loss = foot_slipping_loss.sum() * 2.
        else:
            foot_slipping_loss = foot_slipping_loss.sum() / valid.sum()
        
        loss_dict['Foot_Slipping_Loss'] = foot_slipping_loss * self.foot_slipping_weight
        return loss_dict
    
    def set_eval_mode(self):
        self.eval_mode = True
        self.foot_slipping_weight = 1000.0
        
        