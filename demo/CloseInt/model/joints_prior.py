'''
 @FileName    : interhuman_diffusion.py
 @EditTime    : 2023-10-14 19:05:18
 @Author      : Buzhen Huang
 @Email       : buzhenhuang@outlook.com
 @Description : 
'''
# import os 
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import torch
from torch import nn
from torch.nn import functional as F
from CloseInt.utils.imutils import cam_crop2full, vis_img
from CloseInt.utils.geometry import perspective_projection
from CloseInt.utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix
from einops import rearrange, repeat
import scipy.ndimage.filters as filters

from CloseInt.model.utils import *
from CloseInt.model.blocks import *

from CloseInt.model.stgcn import st_gcn
from CloseInt.model.graph import Graph

class joints_prior(nn.Module):
    def __init__(self, frame_length=16, smooth=False, **kwargs):
        super(joints_prior, self).__init__()

        num_frame = frame_length
        num_agent = 2
        self.eval_initialized = False
        num_timesteps = 100
        beta_scheduler = 'cosine'
        self.timestep_respacing = 'ddim5'

        # Use float64 for accuracy.
        betas = get_named_beta_schedule(beta_scheduler, num_timesteps)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.sampler = UniformSampler(num_timesteps)

        self.cfg_scale = 1.3
        self.num_frames = frame_length
        self.latent_dim = 256
        self.ff_size = self.latent_dim * 2
        self.num_layers = 4
        self.num_heads = 8
        self.dropout = 0.1
        self.activation = 'gelu'
        self.input_feats = 26*3
        self.time_embed_dim = 1024
        img_embed_dim = 1024

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Input Embedding
        self.use_stgcn = True
        if not self.use_stgcn:
            self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        else:
            # Load graph.
            graph_args = {'strategy': 'spatial', 'layout': 'halpe'}
            graph = Graph(**graph_args)
            A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
            self.register_buffer('A', A)
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(A.size()))
            ])
            spatial_kernel_size = A.size(0)
            temporal_kernel_size = 3
            kernel_size = (temporal_kernel_size, spatial_kernel_size)
            self.motion_embed_stgcn = st_gcn(in_channels=3, out_channels=self.latent_dim, kernel_size=kernel_size,
                                             stride=1)
            self.attn = nn.Sequential(
                nn.Linear(self.latent_dim, 1),  # 为每个关节生成注意力权重
                nn.Softmax(dim=3)          # 在关节维度归一化
            )

        self.feature_embed = nn.Linear(self.latent_dim, self.latent_dim)

        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.blocks.append(TransformerBlock(num_heads=self.num_heads,latent_dim=self.latent_dim, dropout=self.dropout, ff_size=self.ff_size))
        # Output Module
        self.out = zero_module(FinalLayer(self.latent_dim + 3, self.input_feats))
        self.smooth = smooth

        self.project = nn.Sequential(
            nn.LayerNorm(img_embed_dim + 3),
            nn.Linear(img_embed_dim + 3, self.latent_dim),
        )

    def condition_process(self, data):
        img_info = {}

        batch_size, frame_length, agent_num = data['features'].shape[:3]

        features = data['features'].reshape(batch_size*frame_length*agent_num, -1)
        # features = torch.zeros_like(features)

        keypoints = data['keypoints']
        pred_keypoints = data['pred_keypoints']
        center = data['center']
        scale = data['scale']
        img_h = data['img_h']
        img_w = data['img_w']
        focal_length = data['focal_length']
        full_img_shape = torch.stack((img_h, img_w), dim=-1)

        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
        # The constants below are used for normalization, and calculated from H36M data.
        # It should be fine if you use the plain Equation (5) in the paper.
        bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)
        # bbox_info = torch.zeros_like(bbox_info)

        cond = torch.cat([features, bbox_info], 1)
        cond = self.project(cond)

        img_info['pred_keypoints'] = pred_keypoints
        img_info['keypoints'] = keypoints
        img_info['bbox_info'] = bbox_info
        img_info['center'] = center
        img_info['scale'] = scale
        img_info['img_h'] = img_h
        img_info['img_w'] = img_w
        img_info['focal_length'] = focal_length
        img_info['full_img_shape'] = full_img_shape
            
        return cond, img_info

    def inference(self, x_t, cond, img_info, data, **kwargs):
            
        batch_size, frame_length, agent_num = data['features'].shape[:3]
        num_valid = batch_size * frame_length * agent_num

        x_a, x_b = x_t[:,:,0], x_t[:,:,1] # [batch_size, frame_length, input_feats], human motion

        mask = None
        if mask is not None:
            mask = mask[...,0]

        emb = self.feature_embed(cond).reshape(-1, self.latent_dim) # [batch_size*frame_length*agent_num, latent_dim], img feature
        emb = emb.reshape(-1, frame_length, agent_num, self.latent_dim)
        emb_a = self.sequence_pos_encoder(emb[:,:,0])
        emb_b = self.sequence_pos_encoder(emb[:,:,1])
        emb = torch.cat([emb_a[:,:,None], emb_b[:,:,None]], dim=2)

        if not self.use_stgcn:
            a_emb = self.motion_embed(x_a) # [batch_size, frame_length, latent_dim]
            b_emb = self.motion_embed(x_b)
        else:
            x_a = x_a.reshape(batch_size, frame_length, 26, 3).permute(0, 3, 1, 2) # [batch_size, 3, frame_length, 26]
            x_b = x_b.reshape(batch_size, frame_length, 26, 3).permute(0, 3, 1, 2)
            a_emb = self.motion_embed_stgcn(x_a, self.A * self.edge_importance[0])
            b_emb = self.motion_embed_stgcn(x_b, self.A * self.edge_importance[0])
            a_emb = a_emb.permute(0, 2, 3, 1) # [batch_size, frame_length, 26, latent_dim]
            b_emb = b_emb.permute(0, 2, 3, 1)
            attn_weights_a = self.attn(a_emb)
            attn_weights_b = self.attn(b_emb)
            a_emb = torch.sum(a_emb * attn_weights_a, dim=2) # [batch_size, frame_length, latent_dim]
            b_emb = torch.sum(b_emb * attn_weights_b, dim=2)

        h_a_prev = self.sequence_pos_encoder(a_emb) # [batch_size, frame_length, latent_dim]
        h_b_prev = self.sequence_pos_encoder(b_emb)

        if mask is None:
            mask = torch.ones(batch_size, frame_length).to(x_a.device)
        key_padding_mask = ~(mask > 0.5)

        counterpart_mask = torch.ones(batch_size, frame_length, 1).to(x_a.device)
        counterpart_mask[data['single_person']>0] = 0.

        for i,block in enumerate(self.blocks):
            h_a = block(h_a_prev, h_b_prev * counterpart_mask, emb[:,:,0], key_padding_mask)
            h_b = block(h_b_prev, h_a_prev * counterpart_mask, emb[:,:,1], key_padding_mask) # (batch_size, frame_length, latent_dim)
            h_a_prev = h_a
            h_b_prev = h_b

        features = torch.cat([h_a[:,:,None], h_b[:,:,None]], dim=2)
        features = features.reshape(-1, self.latent_dim)

        xc = torch.cat([features, img_info['bbox_info']], 1)    

        out = self.out(xc)
        destandardized_joints = out.reshape(batch_size, frame_length, agent_num, 26, 3)

        # destandardize = kwargs.get('destandardize', None)
        # assert destandardize is not None
        # root_pos_init = destandardize['root_pos_init']
        # rot_init = destandardize['rot_init']
        # destandardized_joints = destandardized_joints
        # rot_mat_exp = rot_init.permute(0,2,1).unsqueeze(1).unsqueeze(1).expand(batch_size, frame_length, agent_num, 3, 3)
        # destandardized_joints = torch.bmm(destandardized_joints.reshape(-1, 26, 3), rot_mat_exp.reshape(-1, 3, 3)).reshape(batch_size, frame_length, agent_num, 26, 3)
        # destandardized_joints = destandardized_joints + root_pos_init[:, None, None, None, :]

        if self.smooth and not self.training:
            destandardized_joints = destandardized_joints.detach().cpu().numpy().reshape(batch_size, frame_length, agent_num, 26, 3)
            destandardized_joints = filters.gaussian_filter1d(destandardized_joints, 3, axis=1, mode='nearest').reshape(-1, 26, 3)
            destandardized_joints = torch.tensor(destandardized_joints, dtype=destandardized_joints.dtype, device=destandardized_joints.device)

        img_h = data['img_h']
        img_w = data['img_w']
        focal_length = data['focal_length']
        center = data['center']
        camera_center = torch.stack([img_w/2, img_h/2], dim=-1)
        pred_keypoints_2d = perspective_projection(destandardized_joints.reshape(-1,26,3),
                                                rotation=torch.eye(3, device=destandardized_joints.device).unsqueeze(0).expand(num_valid, -1, -1),
                                                translation=torch.zeros(3, device=destandardized_joints.device).unsqueeze(0).expand(num_valid, -1),
                                                focal_length=focal_length,
                                                camera_center=camera_center)

        pred_keypoints_2d = (pred_keypoints_2d - center[:,None,:]) / 256 #constants.IMG_RES

        pred_cam_t = torch.zeros_like(data['pred_cam_t']) # for compatibility and simplicity

        pred = {
                'pred_joints':destandardized_joints.reshape(-1, 26, 3),
                'pred_cam_t':pred_cam_t.reshape(-1,3), # camera translation
                'focal_length':focal_length,
                'pred_keypoints_2d':pred_keypoints_2d.reshape(-1,26,2), # keypoints in 2D
            }

        return pred


    def forward(self, data):

        batch_size, frame_length, agent_num = data['features'].shape[:3]

        cond, img_info = self.condition_process(data)

        gt_joints = data['pred_joints']
        gt_trans = data['pred_cam_t']
        
        g_joints = gt_joints[...,:3] + gt_trans[..., None,:]
        g_joints = g_joints.reshape(batch_size, frame_length, agent_num, 26, 3)

        pred = self.inference(g_joints, cond, img_info, data)

        return pred