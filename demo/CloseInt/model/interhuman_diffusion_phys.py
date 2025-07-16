'''
 @FileName    : interhuman_diffusion_phys.py
 @EditTime    : 2024-04-15 15:12:19
 @Author      : Buzhen Huang
 @Email       : buzhenhuang@outlook.com
 @Description : 
'''

import torch
import os
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from CloseInt.utils.imutils import cam_crop2full, vis_img, cam_full2crop
from CloseInt.utils.geometry import perspective_projection
from CloseInt.utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix
from einops import rearrange, repeat
import torch.optim as optim

from CloseInt.model.interhuman_diffusion import interhuman_diffusion
from CloseInt.model.interhuman_diffusion_j3d import interhuman_diffusion_j3d
from CloseInt.model.joints_prior import joints_prior
import scipy.ndimage.filters as filters

from CloseInt.model.utils import *
from CloseInt.model.blocks import *

import time
from CloseInt.utils.mesh_intersection.bvh_search_tree import BVH
import CloseInt.utils.mesh_intersection.loss as collisions_loss
from CloseInt.utils.mesh_intersection.filter_faces import FilterFaces

from CloseInt.utils.FileLoaders import load_pkl
from CloseInt.utils.plot_script import draw_keypoints_and_connections
import cv2

# TODO: create only one model for all the tasks
# from CloseInt.model.smpl2capsule import Smpl2Capscule, save_capsule_model, render_capsule_mesh

class interhuman_diffusion_phys(interhuman_diffusion):
    def __init__(self, smpl, use_classifier_guidance=False, scale_factor = 1000000.0, **kwargs):
        super(interhuman_diffusion_phys, self).__init__(smpl, **kwargs)

        # self.use_phys = False
        # self.use_proj_grad = False

        # self.use_interprior = False

        # if self.use_phys:
        #     self.feature_emb_dim = self.feature_emb_dim + 1
        # if self.use_proj_grad:
        #     self.feature_emb_dim = self.feature_emb_dim + 78

        # self.feature_embed = nn.Linear(self.feature_emb_dim, self.latent_dim)

        # if self.use_interprior:
        #     from CloseInt.model.interprior.inter_pose_vqvae import InterHumanTokenizer
        #     self.prior = InterHumanTokenizer(smpl)
        #     check_point = '/root/autodl-tmp/Hi4D/out/interVAE_prior/02.01-14h48m15s/trained model/prior_epoch010.pkl'
        #     model_dict = self.prior.state_dict()
        #     params = torch.load(check_point)
        #     premodel_dict = params['model']
        #     premodel_dict = {k: v for k ,v in premodel_dict.items() if k in model_dict}
        #     model_dict.update(premodel_dict)
        #     self.prior.load_state_dict(model_dict)
        #     # print("Load pretrain parameters from %s" %check_point)
        #     self.prior.requires_grad_(False)
        #     self.prior.eval()

        num_timesteps = 100
        beta_scheduler = 'cosine'
        self.timestep_respacing = 'ddim5'

        self.scale_factor = scale_factor

        # self.search_tree = BVH(max_collisions=8)
        # self.pen_distance = collisions_loss.DistanceFieldPenetrationLoss(sigma=0.0001,
        #                                                  point2plane=False,
        #                                                  vectorized=True)

        self.device = torch.device('cuda')

        # self.part_segm_fn = False #"data/smpl_segmentation.pkl"
        # if self.part_segm_fn:
        #     data = load_pkl(self.part_segm_fn)

        #     faces_segm = data['segm']
        #     ign_part_pairs = [
        #         "9,16", "9,17", "6,16", "6,17", "1,2",
        #         "33,40", "33,41", "30,40", "30,41", "24,25",
        #     ]

        #     # tpose = bm()
        #     # tmp_mesh = trimesh.Trimesh(vertices=tpose.vertices[0].cpu().detach().numpy(), faces=bm.faces)
        #     # min_mask_v = faces_segm.min()
        #     # max_mask_v = faces_segm.max()
        #     # for v in range(min_mask_v, max_mask_v + 1):
        #     #     mmm = tmp_mesh.copy()
        #     #     mmm.visual.face_colors[faces_segm == v, :3] = [0, 255, 0]
        #     #     os.makedirs(osp.join(osp.dirname(__file__), "data/smpl_segm/mask/"), exist_ok=True)
        #     #     mmm.export(osp.join(osp.dirname(__file__), "data/smpl_segm/mask/", "smpl_segm_{}.obj".format(v)))

        #     faces_segm = torch.tensor(faces_segm, dtype=torch.long,
        #                         device=self.device).unsqueeze_(0).repeat([2, 1]) # (2, 13766)

        #     faces_segm = faces_segm + \
        #         (torch.arange(2, dtype=torch.long).to(self.device) * 24)[:, None]
        #     faces_segm = faces_segm.reshape(-1) # (2*13766, )

        #     # faces_parents = torch.tensor(faces_parents, dtype=torch.long,
        #     #                        device=device).unsqueeze_(0).repeat([2, 1]) # (2, 13766)

        #     # faces_parents = faces_parents + \
        #     #     (torch.arange(2, dtype=torch.long).to(device) * 24)[:, None]
        #     # faces_parents = faces_parents.reshape(-1) # (2*13766, )

        #     # Create the module used to filter invalid collision pairs
        #     self.filter_faces = FilterFaces(faces_segm=faces_segm, ign_part_pairs=ign_part_pairs).to(device=self.device)
        self.use_classifier_guidance_cond = False
        self.use_classifier_guidance_mean = False
        self.use_classifier_guidance = use_classifier_guidance
        if self.use_classifier_guidance or self.use_classifier_guidance_mean or self.use_classifier_guidance_cond:
            print('Using classifier guidance')
            
            # load smpl2capsule model for collision loss
            # self.smpl2capsule = Smpl2Capscule()
            # state_dict = torch.load('data/smpl2cap.ckpt')['state_dict']
            # # remove the prefix 'CloseInt.model.'
            # new_state_dict = {}
            # for k, v in state_dict.items():
            #     new_state_dict[k[6:]] = v
            # self.smpl2capsule.load_state_dict(new_state_dict)
            # self.smpl2capsule.to(self.device)
            # self.smpl2capsule.requires_grad_(False)
            # self.smpl2capsule.eval()
            
            # # weight for loss
            # self.consistency_weight = 1.0  # Add a weight for the consistency loss
            # self.interaction_weight = 1.0  # Add a weight for the interaction loss

            self.joint_prior = joints_prior()
            joint_prior_model_path = '/root/autodl-tmp/PW3D/out/refine_joint_prior_pw3d_only/04.23-13h26m34s/trained model/best_reconstruction_j3d_epoch067_209.220245.pkl'
            # state_dict = torch.load('/root/autodl-tmp/Hi4D/out/refine_joints_prior64/04.05-13h51m42s/trained model/best_reconstruction_j3d_epoch055_200.682755.pkl')['model']
            state_dict = torch.load(joint_prior_model_path)['model']
            print('Loading joint prior model from %s' % joint_prior_model_path)
            # remove the prefix 'CloseInt.model.'
            # new_state_dict = {}
            # for k, v in state_dict.items():
            #     new_state_dict[k[6:]] = v
            self.joint_prior.load_state_dict(state_dict)
            self.joint_prior.to(self.device)
            self.joint_prior.requires_grad_(False)
            self.joint_prior.eval()

            # self.contact_prior = np.load('data/mean_contact.npy')
            # self.contact_thresh = 0.05
            # # sparse_smpl is the indices of contact_prior > self.contact_thresh
            # self.sparse_smpl = np.where(self.contact_prior > self.contact_thresh)[0]
            # if len(self.sparse_smpl) > 1000:
            #     # randomly sample 1000 points
            #     self.sparse_smpl = np.random.choice(self.sparse_smpl, 1000, replace=False)
            # self.contact_prior = torch.tensor(self.contact_prior, dtype=torch.float32, device=self.device)[self.sparse_smpl] # [6890, 1] -> [N_sparse, 1]
        
    def visualize(self, pose, shape, pred_cam, data, img_info, t_idx, name='images_phys'):
        import cv2
        from CloseInt.utils.renderer_pyrd import Renderer
        import os
        from CloseInt.utils.FileLoaders import save_pkl
        from CloseInt.utils.module_utils import save_camparam

        # if t_idx not in [0, 5, 10, 15, 20]:
        #     return

        output = os.path.join('test_debug', name)
        os.makedirs(output, exist_ok=True)

        batch_size, frame_length, agent_num = data['features'].shape[:3]

        pose_mtx = rotation_6d_to_matrix(pose.reshape(-1, 6)).view(-1, 24, 3, 3)
        pose = matrix_to_axis_angle(pose_mtx.view(-1, 3, 3)).view(-1, 72)

        # convert the camera parameters from the crop camera to the full camera
        img_h, img_w = img_info['img_h'], img_info['img_w']
        focal_length = img_info['focal_length']
        center = img_info['center']
        scale = img_info['scale']
        
        
        if self.use_cfg_sampler and not self.training:
            img_h = img_h[:num_valid]
            img_w = img_w[:num_valid]
            focal_length = focal_length[:num_valid]
            center = center[:num_valid]
            scale = scale[:num_valid]


        if self.use_cfg_sampler and not self.training:
            img_h = img_h[:num_valid]
            img_w = img_w[:num_valid]
            focal_length = focal_length[:num_valid]
            center = center[:num_valid]
            scale = scale[:num_valid]

        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)
    
        pred_verts, pred_joints = self.smpl(shape, pose, pred_trans, halpe=True)

        shape = shape.reshape(batch_size*frame_length, agent_num, -1).detach().cpu().numpy()
        pose = pose.reshape(batch_size*frame_length, agent_num, -1).detach().cpu().numpy()
        pred_trans = pred_trans.reshape(batch_size*frame_length, agent_num, -1).detach().cpu().numpy()

        pred_verts = pred_verts.reshape(batch_size*frame_length, agent_num, 6890, 3)
        focal_length = focal_length.reshape(batch_size*frame_length, agent_num, -1)[:,0]
        imgs = data['imgname']

        # testing
        vertices = pred_verts.view(batch_size*frame_length*agent_num, -1, 3)
        face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long,
                                device=vertices.device).unsqueeze_(0).repeat([batch_size*frame_length*agent_num,
                                                                        1, 1])
        bs, nv = vertices.shape[:2] # nv: 6890
        bs, nf = face_tensor.shape[:2] # nf: 13776
        faces_idx = face_tensor + (torch.arange(bs, dtype=torch.long).to(vertices.device) * nv)[:, None, None]
        faces_idx = faces_idx.reshape(bs // 2, -1, 3)
        triangles = vertices.view([-1, 3])[faces_idx]

        # print_timings = False
        # with torch.no_grad():
        #     if print_timings:
        #         start = time.time()
        #     collision_idxs = self.search_tree(triangles) # (128, n_coll_pairs, 2)
        #     if print_timings:
        #         torch.cuda.synchronize()
        #         print('Collision Detection: {:5f} ms'.format((time.time() - start) * 1000))

        #     if self.part_segm_fn:
        #         if print_timings:
        #             start = time.time()
        #         collision_idxs = self.filter_faces(collision_idxs)
        #         if print_timings:
        #             torch.cuda.synchronize()
        #             print('Collision filtering: {:5f}ms'.format((time.time() -
        #                                                         start) * 1000))

        # if print_timings:
        #         start = time.time()
        # pen_loss = self.pen_distance(triangles, collision_idxs)
        # if print_timings:
        #     torch.cuda.synchronize()
        #     print('Penetration loss: {:5f} ms'.format((time.time() - start) * 1000))

        pred_verts = pred_verts.detach().cpu().numpy()
        focal_length = focal_length.detach().cpu().numpy()
        # pen_loss = pen_loss.detach().cpu().numpy()

        for index, (img, pred_vert, focal) in enumerate(zip(imgs, pred_verts, focal_length)):
            if index > 15:
                break

            name = img[-40:].replace('\\', '_').replace('/', '_')

            # seq, cam, na = img.split('/')[-3:]
            # if seq != 'sidehug37' or cam != 'Camera64' or na != '000055.jpg':
            #     continue

            img = cv2.imread(img)
            img_h, img_w = img.shape[:2]
            renderer = Renderer(focal_length=focal, center=(img_w/2, img_h/2), img_w=img.shape[1], img_h=img.shape[0], faces=self.smpl.faces, same_mesh_color=True)


            pred_smpl = renderer.render_front_view(pred_vert, bg_img_rgb=img.copy())
            pred_smpl_side = renderer.render_side_view(pred_vert)
            pred_smpl = np.concatenate((img, pred_smpl, pred_smpl_side), axis=1)

            # pred_smpl = cv2.putText(pred_smpl, 'Pen: ' + str(pen_loss[index]), (50,350),cv2.FONT_HERSHEY_COMPLEX,2,(255,191,105),5)

            img_path = os.path.join(output, 'images')
            os.makedirs(img_path, exist_ok=True)
            render_name = "%s_%02d_timestep%02d_pred_smpl.jpg" % (name, index, t_idx)
            cv2.imwrite(os.path.join(img_path, render_name), pred_smpl)

            renderer.delete()

            data = {}
            data['pose'] = pose[index]
            data['trans'] = pred_trans[index]
            data['betas'] = shape[index]

            intri = np.eye(3)
            intri[0][0] = focal
            intri[1][1] = focal
            intri[0][2] = img_w / 2
            intri[1][2] = img_h / 2
            extri = np.eye(4)
            
            cam_path = os.path.join(output, 'camparams', name)
            os.makedirs(cam_path, exist_ok=True)
            save_camparam(os.path.join(cam_path, 'timestep%02d_camparams.txt' %t_idx), [intri], [extri])

            path = os.path.join(output, 'params', name)
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, 'timestep%02d_0000.pkl' %t_idx)
            save_pkl(path, data)

    # def projection_gradients(self, pred_joints, center, img_h, img_w, focal_length, keypoints, eval_idx):
    #     center = center[eval_idx]
    #     img_w = img_w[eval_idx]
    #     img_h = img_h[eval_idx]
    #     focal_length = focal_length[eval_idx]
    #     keypoints = keypoints[eval_idx]
    #     num_valid = pred_joints.shape[0]

    #     center = center.reshape(-1, 2)
    #     img_w = img_w.reshape(-1,)
    #     img_h = img_h.reshape(-1,)
    #     focal_length = focal_length.reshape(-1,)
    #     keypoints = keypoints.reshape(-1, 26, 3)

    #     with torch.enable_grad():
    #         pred_joints = pred_joints.detach().requires_grad_(True)
    #         camera_center = torch.stack([img_w/2, img_h/2], dim=-1)
    #         pred_keypoints_2d = perspective_projection(pred_joints,
    #                                                 rotation=torch.eye(3, device=pred_joints.device).unsqueeze(0).expand(num_valid, -1, -1),
    #                                                 translation=torch.zeros(3, device=pred_joints.device).unsqueeze(0).expand(num_valid, -1),
    #                                                 focal_length=focal_length,
    #                                                 camera_center=camera_center)

    #         pred_keypoints_2d = (pred_keypoints_2d - center[:,None,:]) / 256

    #         loss = torch.sqrt(torch.sum((pred_keypoints_2d - keypoints[...,:2])**2, dim=-1) * keypoints[...,2])
    #         loss = loss.mean()

    #         grad = torch.autograd.grad(-loss, pred_joints)[0]

    #     vis = False
    #     if vis:
    #         import cv2
    #         for gt_kp, pred_kp, w, h, c in zip(keypoints.detach().cpu().numpy(), pred_keypoints_2d.detach().cpu().numpy(), img_w.detach().cpu().numpy(), img_h.detach().cpu().numpy(), center.detach().cpu().numpy()):
    #             gt_kp = gt_kp[:,:2] * 256 + c
    #             pred_kp = pred_kp[:,:2] * 256 + c
    #             img = np.zeros((int(h), int(w), 3), dtype=np.int8)
    #             for kp in gt_kp.astype(np.int):
    #                 img = cv2.circle(img, tuple(kp), 5, (0,0,255), -1)

    #             for kp in pred_kp.astype(np.int):
    #                 img = cv2.circle(img, tuple(kp), 5, (0,255,255), -1)

    #             vis_img('img', img)
        
    #     return grad.detach()

    # def eval_physical_plausibility(self, t, x_t, mean, img_info):

    #     pen_losses = -1 * torch.ones((x_t.shape[0], x_t.shape[1], 1), dtype=x_t.dtype, device=x_t.device)

    #     eval_idx = t < self.num_timesteps * 0.2

    #     x_t = x_t + mean

    #     batch_size, frame_length, agent_num = x_t.shape[:3]

    #     if eval_idx.sum() < 1:
    #         pen_losses = pen_losses.repeat(1, 1, agent_num)
    #         pen_losses = pen_losses.reshape(-1, 1)

    #         proj_gradients = torch.zeros((batch_size, frame_length, agent_num, 26, 3), dtype=x_t.dtype, device=x_t.device)
    #         proj_gradients = proj_gradients.reshape(-1, 26*3)

    #         return pen_losses, proj_gradients

    #     pose = x_t[:,:,:,:144].reshape(-1, 144).contiguous()
    #     shape = x_t[:,:,:,144:154].reshape(-1, 10).contiguous()
    #     pred_cam = x_t[:,:,:,154:].reshape(-1, 3).contiguous()

    #     pose_mtx = rotation_6d_to_matrix(pose.reshape(-1, 6)).view(-1, 24, 3, 3)
    #     pose = matrix_to_axis_angle(pose_mtx.view(-1, 3, 3)).view(-1, 72)

    #     # convert the camera parameters from the crop camera to the full camera
    #     img_h, img_w = img_info['img_h'], img_info['img_w']
    #     focal_length = img_info['focal_length']
    #     center = img_info['center']
    #     scale = img_info['scale']

    #     full_img_shape = torch.stack((img_h, img_w), dim=-1)
    #     pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)

    #     shape = shape.reshape(batch_size, frame_length, agent_num, -1)
    #     pose = pose.reshape(batch_size, frame_length, agent_num, -1)
    #     pred_trans = pred_trans.reshape(batch_size, frame_length, agent_num, -1)

    #     if self.use_proj_grad:
    #         center = center.reshape(batch_size, frame_length, agent_num, -1)
    #         img_h = img_h.reshape(batch_size, frame_length, agent_num, -1)
    #         img_w = img_w.reshape(batch_size, frame_length, agent_num, -1)
    #         focal_length = focal_length.reshape(batch_size, frame_length, agent_num, -1)
            
    #     keypoints = img_info['keypoints'].reshape(batch_size, frame_length, agent_num, -1, 3)

    #     shape = shape[eval_idx]
    #     pose = pose[eval_idx]
    #     pred_trans = pred_trans[eval_idx]

    #     batch_size, frame_length, agent_num = shape.shape[:3]

    #     pose = pose.reshape(-1, 72).contiguous()
    #     shape = shape.reshape(-1, 10).contiguous()
    #     pred_trans = pred_trans.reshape(-1, 3).contiguous()

    #     pred_verts, pred_joints = self.smpl(shape, pose, pred_trans, halpe=True)

    #     # projection gradients
    #     if self.use_proj_grad:
    #         proj_gradients = torch.zeros_like(keypoints)
    #         proj_grad = self.projection_gradients(pred_joints, center, img_h, img_w, focal_length, keypoints, eval_idx)
    #         proj_grad = pred_joints + 100 * proj_grad 
    #         proj_grad = proj_grad.reshape(batch_size, frame_length, agent_num, 26, 3)
    #         proj_gradients[eval_idx] = proj_grad
    #         proj_gradients = proj_gradients.reshape(-1, 26*3)
    #     else:
    #         proj_gradients = torch.zeros_like(keypoints)
    #         proj_gradients = proj_gradients.reshape(-1, 26*3)

    #     if self.use_phys:

    #         # testing
    #         vertices = pred_verts.view(batch_size*frame_length*agent_num, -1, 3)
    #         face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long,
    #                                 device=vertices.device).unsqueeze_(0).repeat([batch_size*frame_length*agent_num,
    #                                                                         1, 1])
    #         bs, nv = vertices.shape[:2] # nv: 6890
    #         bs, nf = face_tensor.shape[:2] # nf: 13776
    #         faces_idx = face_tensor + (torch.arange(bs, dtype=torch.long).to(vertices.device) * nv)[:, None, None]
    #         faces_idx = faces_idx.reshape(bs // 2, -1, 3)
    #         triangles = vertices.view([-1, 3])[faces_idx]

    #         print_timings = False
    #         with torch.no_grad():
    #             if print_timings:
    #                 start = time.time()
    #             collision_idxs = self.search_tree(triangles) # (128, n_coll_pairs, 2)
    #             if print_timings:
    #                 torch.cuda.synchronize()
    #                 print('Collision Detection: {:5f} ms'.format((time.time() - start) * 1000))

    #             if self.part_segm_fn:
    #                 if print_timings:
    #                     start = time.time()
    #                 collision_idxs = self.filter_faces(collision_idxs)
    #                 if print_timings:
    #                     torch.cuda.synchronize()
    #                     print('Collision filtering: {:5f}ms'.format((time.time() -
    #                                                                 start) * 1000))

    #         if print_timings:
    #                 start = time.time()
    #         pen_loss = self.pen_distance(triangles, collision_idxs)
    #         if print_timings:
    #             torch.cuda.synchronize()
    #             print('Penetration loss: {:5f} ms'.format((time.time() - start) * 1000))

    #         pen_loss = pen_loss.reshape(batch_size, frame_length, -1)

    #         pen_losses[eval_idx] = pen_loss / 1000.
    #         pen_losses = pen_losses.repeat(1, 1, agent_num)
    #         pen_losses = pen_losses.reshape(-1, 1)

    #     else:
    #         pen_losses = pen_losses.repeat(1, 1, agent_num)
    #         pen_losses = pen_losses.reshape(-1, 1)

    #     return pen_losses, proj_gradients

    def interprior(self, t, x_t, data, img_info):

        batch_size, frame_length, agent_num = data['features'].shape[:3]

        # eval_idx = t < self.num_timesteps * 0.2

        # if eval_idx.sum() < 1:
        #     return x_t

        pose = x_t[:,:,:,:144].reshape(-1, 144).contiguous()
        shape = x_t[:,:,:,144:154].reshape(-1, 10).contiguous()
        pred_cam = x_t[:,:,:,154:].reshape(-1, 3).contiguous()

        viz = False
        if viz:
            self.visualize(pose, shape, pred_cam, data, img_info, int(t[0].detach().cpu().numpy()), name='before_prior')

        # pose_mtx = rotation_6d_to_matrix(pose.reshape(-1, 6)).view(-1, 24, 3, 3)
        # pose_aa = matrix_to_axis_angle(pose_mtx.view(-1, 3, 3)).view(-1, 72)

        # convert the camera parameters from the crop camera to the full camera
        img_h, img_w = img_info['img_h'], img_info['img_w']
        focal_length = img_info['focal_length']
        center = img_info['center']
        scale = img_info['scale']

        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)

        shape = shape.reshape(batch_size, frame_length, agent_num, -1)
        pose = pose.reshape(batch_size, frame_length, agent_num, -1)
        pred_trans = pred_trans.reshape(batch_size, frame_length, agent_num, -1)
        
        trans_a = pred_trans[:,:,0,:]
        r_trans = pred_trans - trans_a[:,:,None,:]
        x = torch.cat([pose,shape,r_trans], dim=-1) # [bs, seq_len, num_person, 144+10+3]
        # Encode
        x_encoder = self.prior.encoder(x)
        ## quantization
        x_quantized, loss, perplexity  = self.prior.quantizer(x_encoder)
        ## decoder
        x_decoder = self.prior.decoder(x_quantized)
        
        updated_pose_6d = x_decoder['pred_pose6d'].reshape(batch_size, frame_length, agent_num, 144)
        updated_shape = x_decoder['pred_shape'].reshape(batch_size, frame_length, agent_num, 10)
        updated_r_trans = x_decoder['pred_cam_rt'].reshape(batch_size, frame_length, agent_num, 3)
        updated_trans = updated_r_trans + trans_a[:,:,None,:]
        
        updadted_cam = cam_full2crop(updated_trans.reshape(-1,3), center, scale, full_img_shape, focal_length)
        updadted_cam = updadted_cam.reshape(batch_size, frame_length, agent_num, 3)

        # center = center.reshape(batch_size, frame_length, agent_num, 2)[eval_idx].reshape(-1,2)
        # scale = scale.reshape(batch_size, frame_length, agent_num)[eval_idx].reshape(-1,)
        # full_img_shape = full_img_shape.reshape(batch_size, frame_length, agent_num, 2)[eval_idx].reshape(-1,2)
        # focal_length = focal_length.reshape(batch_size, frame_length, agent_num)[eval_idx].reshape(-1,)
        # updated_trans = updated_trans.reshape(-1, 3)

        # updated_cam = cam_full2crop(updated_trans, center, scale, full_img_shape, focal_length)

        # updated_cam = updated_cam.reshape(-1, frame_length, agent_num, 3)

        x_t_updated = torch.cat([updated_pose_6d, updated_shape, updadted_cam], dim=-1)

        x_t = x_t_updated

        viz = False
        if viz:
            pose = x_t[:,:,:,:144].reshape(-1, 144).contiguous()
            shape = x_t[:,:,:,144:154].reshape(-1, 10).contiguous()
            pred_cam = x_t[:,:,:,154:].reshape(-1, 3).contiguous()

            self.visualize(pose, shape, pred_cam, data, img_info, int(t[0].detach().cpu().numpy()), name='after_prior')

        return x_t_updated

    def forward(self, data):

        batch_size, frame_length, agent_num = data['features'].shape[:3]
        num_valid = batch_size * frame_length * agent_num

        cond, img_info = self.condition_process(data)

        init_pose = data['init_pose_6d']
        if not self.training and 'comp_pose_6d' in data:
            gt_cam_t = data['gt_cam_t'].reshape(batch_size, frame_length, agent_num, 3)
            comp_pose_6d = data['comp_pose_6d'].reshape_as(data['init_pose_6d']) # [bs, seq_len, num_person, 144]
            for b in range(batch_size):
                for f in range(frame_length):
                    init_pose_6d_a = init_pose[b, f, 0]
                    init_pose_6d_b = init_pose[b, f, 1]
                    depth_a = gt_cam_t[b, f, 0, 2]
                    depth_b = gt_cam_t[b, f, 1, 2]
                    further_agent = 0 if depth_a > depth_b else 1
                    if torch.norm(init_pose_6d_a - init_pose_6d_b) < 0.1:
                        print('Pose is the same for', data['imgname'][b*frame_length+f])
                        init_pose[b, f, further_agent] = comp_pose_6d[b, f, further_agent]
        noise, mean = self.generate_noise(init_pose)

        if self.training:
            
            x_start = self.input_process(data, img_info, mean)

            t, _ = self.sampler.sample(batch_size, x_start.device)

            # visualization
            viz_sampling = False
            if viz_sampling:
                self.visualize_sampling(x_start, t, data, img_info, mean, noise=noise)

            x_t = self.q_sample(x_start, t, noise=noise)

            # if self.use_phys or self.use_proj_grad:
            #     pen_loss, proj_grads = self.eval_physical_plausibility(t, x_t, mean, img_info)
            #     proj_grads = img_info['pred_keypoints'].reshape(-1, 78)
            #     if self.use_phys:
            #         cond = torch.cat([cond, pen_loss], dim=1)
            #     if self.use_proj_grad:
            #         cond = torch.cat([cond, proj_grads], dim=1)

            pred = self.inference(x_t, t, cond, img_info, data, mean)

        else:
            if not self.eval_initialized:
                self.init_eval()
                self.eval_initialized = True
                
            pred = self.ddim_sample_loop(noise, mean, cond, img_info, data)
            # init_pose_6d = data['init_pose_6d'].reshape(batch_size, frame_length, agent_num, 144)
            # pred_pose_6d = pred['pred_pose6d'].reshape_as(data['init_pose_6d']) # [bs, seq_len, num_person, 144]
            # # pose_6d = data['pose_6d'].reshape(batch_size, frame_length, agent_num, 144)
            # # # for b in range(batch_size):
            # # #     for f in range(frame_length):
            # # #         init_pose_6d_a = init_pose_6d[b, f, 0]
            # # #         init_pose_6d_b = init_pose_6d[b, f, 1]
            # # #         if torch.norm(init_pose_6d_a - init_pose_6d_b) < 0.1:
            # # #             print('Pose is the same for', data['imgname'][b*frame_length+f])
            # # #             init_pose_6d[b, f] = pred_pose_6d[b, f]
            # _, mean = self.generate_noise(pred_pose_6d)
            # pred = self.ddim_sample_loop(noise, mean, cond, img_info, data)


        return pred
    
    def get_target_joints_from_affordance(self, affordance, pred_g_joints, pred_cam_t, data):
        if affordance.max() < 0.5:
            return pred_g_joints

        batch_size, frame_length, agent_num = data['features'].shape[:3]
        pred_cam_t = pred_cam_t.reshape(batch_size, frame_length, agent_num, 3)
        pred_g_joints = pred_g_joints.reshape(batch_size, frame_length, agent_num, 26, 3)
        # depth_mean for each batch
        depth_a = pred_cam_t[:,:,0,2].mean(dim=-1)
        depth_b = pred_cam_t[:,:,1,2].mean(dim=-1)

        closer_agent = torch.where(depth_a < depth_b, 0, 1) # [bs]

        joints_affordace = torch.tensordot(affordance.reshape(-1,6890,1), self.smpl.J_halpe_regressor.transpose(0, 1), dims=([1], [0])).transpose(1, 2)
        joints_affordace = joints_affordace.reshape(batch_size, frame_length, agent_num, 26)
        contact_mask = joints_affordace > 0.1 # [bs, seq_len, num_person, 26]

        target_joint = pred_g_joints.clone()
        masked_joint = pred_g_joints * contact_mask.unsqueeze(-1).float()
        j_dis_mat = torch.cdist(masked_joint[:,:,0], masked_joint[:,:,1], p=2) # [bs, seq_len, 26, 26]
        closest_joint_ind_a = j_dis_mat.argmin(dim=-1) # [bs, seq_len, 26], joint index of agnet b that is closest to agent a's 26 joints
        closest_joint_ind_b = j_dis_mat.argmin(dim=-2) # [bs, seq_len, 26], joint index of agnet a that is closest to agent b's 26 joints
        closest_joint_ind = torch.cat([closest_joint_ind_a.unsqueeze(2), closest_joint_ind_b.unsqueeze(2)], dim=2) # [bs, seq_len, 26*2]
        
        for b in range(batch_size):
            for f in range(frame_length):
                for j in range(26):
                    if contact_mask[b, f, 1-closer_agent[b], j]:
                        target_joint[b, f, 1-closer_agent[b], j] = pred_g_joints[b, f, closer_agent[b], closest_joint_ind[b, f, 1-closer_agent[b], j]]
                    if contact_mask[b, f, closer_agent[b], j]:
                        target_joint[b, f, 1-closer_agent[b], closest_joint_ind[b, f, closer_agent[b], j]] = pred_g_joints[b, f, closer_agent[b], j]

        return target_joint
    
    def get_foot_slipping_loss(self, pred_joints, data, img_info, debug=False, t=0, i=0):
        """
        args:
            pred_joint: [bs, seq_len, num_person, 24, 3]
            data: dict
            img_info: dict
        return:
            foot_slipping_loss: [bs, seq_len, num_person, 2]
            foot_slipping_loss_mean:
        """
        batch_size, frame_length, agent_num = data['features'].shape[:3]
        pred_joints = pred_joints.reshape(batch_size, frame_length, agent_num, 24, 3)
        # lower foot should not move
        lfoot_ids = [7, 10]
        rfoot_ids = [8, 11]

        pred_joints_a = pred_joints[:,:,0] # [bs, seq_len, 24, 3]
        pred_joints_b = pred_joints[:,:,1]

        lfoot_vel_a = pred_joints_a[..., 1:, lfoot_ids,:] - pred_joints_a[..., :-1, lfoot_ids,:] # [bs, seq_len-1, 2, 3]
        lfoot_vel_a = lfoot_vel_a.mean(dim=-2) # [bs, seq_len-1, 3]
        rfoot_vel_a = pred_joints_a[..., 1:, rfoot_ids,:] - pred_joints_a[..., :-1, rfoot_ids,:]
        rfoot_vel_a = rfoot_vel_a.mean(dim=-2)

        lfoot_h_a = pred_joints_a[..., :-1, lfoot_ids, 1].mean(dim=-1) # [bs, seq_len-1]
        rfoot_h_a = pred_joints_a[..., :-1, rfoot_ids, 1].mean(dim=-1)

        lfoot_slip_a = torch.where(lfoot_h_a < rfoot_h_a, lfoot_vel_a.norm(dim=-1), torch.zeros_like(lfoot_vel_a.norm(dim=-1))) # [bs, seq_len]
        rfoot_slip_a = torch.where(rfoot_h_a < lfoot_h_a, rfoot_vel_a.norm(dim=-1), torch.zeros_like(rfoot_vel_a.norm(dim=-1))) # [bs, seq_len]

        lfoot_vel_b = pred_joints_b[..., 1:, lfoot_ids,:] - pred_joints_b[..., :-1, lfoot_ids,:]
        lfoot_vel_b = lfoot_vel_b.mean(dim=-2)
        rfoot_vel_b = pred_joints_b[..., 1:, rfoot_ids,:] - pred_joints_b[..., :-1, rfoot_ids,:]
        rfoot_vel_b = rfoot_vel_b.mean(dim=-2)

        lfoot_h_b = pred_joints_b[..., :-1, lfoot_ids, 1].mean(dim=-1)
        rfoot_h_b = pred_joints_b[..., :-1, rfoot_ids, 1].mean(dim=-1)

        lfoot_slip_b = torch.where(lfoot_h_b < rfoot_h_b, lfoot_vel_b.norm(dim=-1), torch.zeros_like(lfoot_vel_b.norm(dim=-1))) # [bs, seq_len]
        rfoot_slip_b = torch.where(rfoot_h_b < lfoot_h_b, rfoot_vel_b.norm(dim=-1), torch.zeros_like(rfoot_vel_b.norm(dim=-1))) # [bs, seq_len]
        
        foot_slipping_loss = lfoot_slip_a + rfoot_slip_a + lfoot_slip_b + rfoot_slip_b
        foot_slipping_loss_mean = foot_slipping_loss.mean()
        return foot_slipping_loss, foot_slipping_loss_mean
    
    def get_projection_loss(self, pred_joints_halpe, data, img_info, debug=False, t=0, i=0):
        """
        args:
            pred_joint: [bs, seq_len, num_person, 24, 3]
            data: dict
            img_info: dict
        return:
            projection_loss: [bs, seq_len, num_person, 26, 2]
            projection_loss_mean:
        """
        batch_size, frame_length, agent_num = data['features'].shape[:3]
        num_valid = batch_size * frame_length * agent_num
        
        img_h, img_w = img_info['img_h'], img_info['img_w']
        focal_length = img_info['focal_length']
        center = img_info['center']
        
        if self.use_cfg_sampler and not self.training:
            img_h = img_h[:num_valid]
            img_w = img_w[:num_valid]
            focal_length = focal_length[:num_valid]
            center = center[:num_valid]
            
        camera_center = torch.stack([img_w/2, img_h/2], dim=-1)
        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        loss_fn = torch.nn.MSELoss(reduction='none') # return square error
        pred_keypoints_2d = perspective_projection(pred_joints_halpe,
                                    rotation=torch.eye(3, device=pred_joints_halpe.device).unsqueeze(0).expand(img_h.shape[0], -1, -1),
                                    translation=torch.zeros(3, device=pred_joints_halpe.device).unsqueeze(0).expand(img_h.shape[0], -1),
                                    focal_length=focal_length,
                                    camera_center=camera_center)  # [bs*seq_len*num_person, 26, 2]
        
        save_kpt = False
        if save_kpt:
            b = 11
            f = 13
            # test_keypoints = pred_keypoints_2d[0].reshape(26, 2).detach().cpu().numpy()
            test_keypoints = pred_keypoints_2d[b*frame_length*2+f*2].reshape(26, 2).detach().cpu().numpy()
            # test_img_name = data['imgname'][0]
            test_img_name = data['imgname'][b*frame_length+f]
            test_img = cv2.imread(test_img_name)
            pred_2d = draw_keypoints_and_connections(test_img, test_keypoints)
            # put text on the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            if debug:
                cv2.putText(pred_2d, f"aft guidance", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(pred_2d, f"before guidance", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            psudo_gt_kpts = data['keypoints'][..., :2]
            psudo_gt_kpts = psudo_gt_kpts * 256 + center[:,None,:]
            test_keypoints = psudo_gt_kpts[b*frame_length*2+f*2].reshape(26, 2).detach().cpu().numpy()
            gt_2d = draw_keypoints_and_connections(test_img, test_keypoints)
            cv2.putText(gt_2d, f"psudo gt", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            kpt_img = np.concatenate([pred_2d, gt_2d], axis=1)
            seq = test_img_name.split('/')[-3]
            camera_seq = test_img_name.split('/')[-2]
            frame_num = test_img_name.split('/')[-1].split('.')[0]
            cv2.imwrite(f'kpt_img_{seq}_{camera_seq}_{frame_num}_{t[0,0,0]}_{i}_{debug}.jpg', kpt_img)
        
        pred_keypoints_2d = (pred_keypoints_2d - center[:,None,:]) / 256
        pred_keypoints_2d = pred_keypoints_2d.reshape(-1, frame_length, agent_num, 26, 2)
        psudo_gt_kpts = data['pred_keypoints'].reshape(batch_size, frame_length, agent_num, 26, 3)[..., :2]

        # ignore the 18th joints
        pred_keypoints_2d = torch.cat([pred_keypoints_2d[..., :17, :], pred_keypoints_2d[..., 18:, :]], dim=-2)
        psudo_gt_kpts = torch.cat([psudo_gt_kpts[..., :17, :], psudo_gt_kpts[..., 18:, :]], dim=-2)
        projection_loss = loss_fn(pred_keypoints_2d, psudo_gt_kpts) # [bs, seq_len, num_person, 26, 2]
        # print(projection_loss.max()*256)
        # print(projection_loss.min()*256)
        # igonre loss that is too large
        projection_loss = torch.where(projection_loss < 0.1, projection_loss, torch.zeros_like(projection_loss))
        
        proj_mean_loss = projection_loss.mean()
        return projection_loss, proj_mean_loss
    
    def cal_interaction_loss(self, caps_dist_mtx):
        """
        Compute the contact loss based on the distance between joints.
        
        Args:
            caps_dist_mtx: Tensor of shape [bs, seq_len, c1, c2] containing distances between joints.        
        Returns:
            attraction_loss: Tensor of shape [bs, seq_len, c1, c2] containing attraction loss.
            repulsion_loss: Tensor of shape [bs, seq_len, c1, c2] containing repulsion loss.
            contact_loss: Scalar containing the total contact loss.
        """
        repulsion_loss = torch.where(caps_dist_mtx < 0, caps_dist_mtx**2, torch.zeros_like(caps_dist_mtx)) # [bs, seq_len, c1, c2]
        
        miu = 0.1
        sigma = 0.025
        attraction_loss = torch.where(caps_dist_mtx > 0, 0.0005 / (sigma * torch.sqrt(2 * torch.tensor(3.1415926))) * torch.exp(-0.5 * ((caps_dist_mtx - miu) / sigma) ** 2), torch.zeros_like(caps_dist_mtx)) # [bs, seq_len, c1, c2]
    
        interaction_loss = repulsion_loss + attraction_loss # [bs, seq_len, c1, c2]
        
        return attraction_loss, repulsion_loss, interaction_loss.mean()
    
    def cal_contact_prior_loss(self, verts_a, verts_b, threshold=0.3):
        """
        当两位角色对应的某些顶点距离小于threshold时，鼓励其与 contact_prior 靠近。
        Args:
            verts_a: [B, F, N, 3], 角色A的顶点坐标
            verts_b: [B, F, N, 3], 角色B的顶点坐标
            contact_prior: [N, ], 预先统计好的每个顶点的接触概率
            threshold: float, 小于该距离时认为顶点存在接触
        Returns:
            contact_loss: 在接触区域与 contact_prior 的差异
        """
        # get bbox for verts at each frame
        min_a, max_a = verts_a.min(dim=2)[0], verts_a.max(dim=2)[0] # [B, F, 3], [B, F, 3]
        min_b, max_b = verts_b.min(dim=2)[0], verts_b.max(dim=2)[0] # [B, F, 3], [B, F, 3]
        
        # check for bbox intersection at each frame
        intersect = (min_a < max_b) & (max_a > min_b) # [B, F, 3]
        intersect = intersect.all(dim=2) # [B, F]
        
        if not intersect.any():
            return torch.tensor(0., dtype=torch.float32, device=verts_a.device)
        
        # filter out frames where bbox does not intersect
        verts_a = verts_a[intersect] # [num_frame has intersection, N, 3]
        verts_b = verts_b[intersect]
        
        verts_a = verts_a[:,self.sparse_smpl,:]
        verts_b = verts_b[:,self.sparse_smpl,:]
        
        num = verts_a.shape[0]
        
        contact_masks = []
        dists = []
        for i in range(num):
            dist = torch.cdist(
                verts_a[i].reshape(-1, 3),
                verts_b[i].reshape(-1, 3)
            ) # [N_sparse, N_sparse]
            dists.append(dist)
            contact_masks.append(dist < threshold)
            
        contact_masks = torch.stack(contact_masks, dim=0) # [num_frame has intersection, N_sparse, N_sparse]
        dists = torch.stack(dists, dim=0) # [num_frame has intersection, N_sparse, N_sparse]
        
        contact_prior_mask = self.contact_prior > self.contact_thresh # [N_sparse]
        # convert to float32
        contact_prior_mask = contact_prior_mask.to(torch.float32)
        contact_prior_mask = torch.matmul(contact_prior_mask.unsqueeze(0), contact_prior_mask.unsqueeze(1)) # [N_sparse, N_sparse]
        contact_prior_mask = contact_prior_mask.unsqueeze(0).expand(num, -1, -1) # [num_frame has intersection, N_sparse, N_sparse]
        combined_mask = contact_masks * contact_prior_mask
    
        contact_loss = dists * combined_mask # [num_frame has intersection, N_sparse, N_sparse]        

        return contact_loss.mean()

    def get_interpolate_frame_penetration_loss(self, shape, pred_joints, pred_joints_halpe, data, img_info, interpolate_frame = 2, debug=False, t=0, i=0):
        """
        args:
            shape: [bs*seq_len*num_person, 10]
            pred_joints: [bs*seq_len*num_person, 24, 3]
            pred_joints_halpe: [bs*seq_len*num_person, 26, 3]
        return:
            interpolate_frame_interaction_loss_mean: float
        """
        batch_size, frame_length, agent_num = data['features'].shape[:3]
        
        # interpolate the joints between two frames
        pred_joints = pred_joints.reshape(batch_size, frame_length, agent_num, 24, 3)
        pred_joints_halpe = pred_joints_halpe.reshape(batch_size, frame_length, agent_num, 26, 3)
        shape = shape.reshape(batch_size, frame_length, agent_num, -1)

        pred_joints_interpolate = []
        pred_joints_halpe_interpolate = []
        shape_interpolate = []

        for i in range(frame_length - 1):
            pred_joints_interpolate.append(pred_joints[:, i])
            pred_joints_halpe_interpolate.append(pred_joints_halpe[:, i])
            shape_interpolate.append(shape[:, i])            
            
            interpolated_joints = (pred_joints[:, i] + pred_joints[:, i + 1]) / 2
            interpolated_joints_halpe = (pred_joints_halpe[:, i] + pred_joints_halpe[:, i + 1]) / 2
            interpolated_shape = (shape[:, i] + shape[:, i + 1]) / 2
            
            pred_joints_interpolate.append(interpolated_joints)
            pred_joints_halpe_interpolate.append(interpolated_joints_halpe)
            shape_interpolate.append(interpolated_shape)

        pred_joints_interpolate.append(pred_joints[:, -1])
        pred_joints_halpe_interpolate.append(pred_joints_halpe[:, -1])
        shape_interpolate.append(shape[:, -1])

        pred_joints_interpolate = torch.stack(pred_joints_interpolate, dim=1).reshape(-1, 24, 3)
        pred_joints_halpe_interpolate = torch.stack(pred_joints_halpe_interpolate, dim=1).reshape(-1, 26, 3)
        shape_interpolate = torch.stack(shape_interpolate, dim=1).reshape(-1, 10)
        
        capsules_pred_interpolate = self.smpl2capsule(shape_interpolate, pred_joints_interpolate, pred_joints_halpe_interpolate)
        capsules_pred_interpolate = capsules_pred_interpolate.reshape(batch_size, 2*frame_length-1, agent_num, self.smpl2capsule.total_capsule_num, 8)
        # Get capsules for each person
        caps_pred_person1 = capsules_pred_interpolate[:, :, 0]  # [bs, seq_len, num_capsules, 8]
        caps_pred_person2 = capsules_pred_interpolate[:, :, 1]  # [bs, seq_len, num_capsules, 8]
        
        # Compute distances between all capsule pairs
        caps_dist_mtx = self.smpl2capsule.compute_capsule_distance(caps_pred_person1, caps_pred_person2)  # [bs, seq_len, c1, c2]
        
        attraction_loss, repulsion_loss, interaction_loss_mean = self.cal_interaction_loss(caps_dist_mtx)
        
        return repulsion_loss.mean()
    
    def get_interaction_loss(self, shape, pred_joints, pred_joints_halpe, data, ignore_hand=False, return_interaction_loss = False, debug=False, t=0, i=0):
        """
        args:
            shape: [bs*seq_len*num_person, 10]
            pred_joints: [bs*seq_len*num_person, 24, 3]
            pred_joints_halpe: [bs*seq_len*num_person, 26, 3]
        return:
            collisions_dist: [bs, seq_len, c1, c2]
            joints_dist: [bs, seq_len, c1, c2]
            collisions_loss: float
            interaction_loss: float
        """
        batch_size, frame_length, agent_num = data['features'].shape[:3]
        
        capsules_pred = self.smpl2capsule(shape, pred_joints, pred_joints_halpe)
        capsules_pred = capsules_pred.reshape(-1, frame_length, agent_num, self.smpl2capsule.total_capsule_num, 8)
                    
        # Get capsules for each person
        caps_pred_person1 = capsules_pred[:, :, 0]  # [bs, seq_len, num_capsules, 8]
        caps_pred_person2 = capsules_pred[:, :, 1]  # [bs, seq_len, num_capsules, 8]
        
        if ignore_hand:
            hand_cap_id = [8,21]
            caps_pred_person1 = torch.cat([caps_pred_person1[..., :hand_cap_id[0], :], caps_pred_person1[..., hand_cap_id[0]+1:hand_cap_id[1], :], caps_pred_person1[..., hand_cap_id[1]+1:, :]], dim=-2) # [bs, seq_len, num_capsules-2, 8]
            caps_pred_person2 = torch.cat([caps_pred_person2[..., :hand_cap_id[0], :], caps_pred_person2[..., hand_cap_id[0]+1:hand_cap_id[1], :], caps_pred_person2[..., hand_cap_id[1]+1:, :]], dim=-2)
        
        # Compute distances between all capsule pairs
        caps_dist_mtx = self.smpl2capsule.compute_capsule_distance(caps_pred_person1, caps_pred_person2)  # [bs, seq_len, c1, c2]
        
        attraction_loss, repulsion_loss, interaction_loss_mean = self.cal_interaction_loss(caps_dist_mtx)
        
        save_stl = False
        if save_stl:
            # save capsules that have collision
            for b in range(capsules_pred.shape[0]):
                if save_stl:
                    for f in range(capsules_pred.shape[1]):
                        if save_stl:
                            for c1 in range(capsules_pred.shape[3]):
                                if save_stl:
                                    for c2 in range(capsules_pred.shape[3]):
                                        if save_stl:
                                            if torch.abs(caps_dist_mtx[b,f,c1,c2]) < 0.01:
                                                img_name = data['imgname'][b*frame_length+f]
                                                seq = img_name.split('/')[-3]
                                                camera_seq = img_name.split('/')[-2]
                                                frame_num = img_name.split('/')[-1].split('.')[0]
                                                test_capsules=capsules_pred[b,f].reshape(-1, self.smpl2capsule.total_capsule_num, 8) # 
                                                # save_capsule_model(test_capsules, 'test_capsules.txt')
                                                render_capsule_mesh(test_capsules, f'test_capsules_{seq}_{camera_seq}_{frame_num}_{c1}_{c2}_{t}_{i}_{debug}.stl')
                                                save_stl = False
                                                
        joints_dist = torch.relu(caps_dist_mtx) # distance between capsules that do not have intersection
        return joints_dist, attraction_loss, repulsion_loss, interaction_loss_mean

    def classifier_guidance(self, x_t, data, img_info, t=0, i=0, debug=False):
        """
        args:
            x_t: [bs, seq_len, num_person, 154]
            data: dict
            img_info: dict
        return:
            x_t: [bs, seq_len, num_person, 154] x_t under guidance if the new loss is smaller
        """
        # print("==========classifier_guidance==========")
        batch_size, frame_length, agent_num = data['features'].shape[:3]
        num_valid = batch_size * frame_length * agent_num
        # orient_6d = x_t[:,:,:,:6]
        pose_6d = x_t[:,:,:,:144].reshape(-1, 6)
        shape = x_t[:,:,:,144:154].reshape(-1, 10)
        pred_cam = x_t[:,:,:,154:].reshape(-1, 3)

        j3d_diff_res = self.joint_prior(data)
        target_g_joints = j3d_diff_res['pred_joints']
        loss_fn = torch.nn.MSELoss(reduction='none')

        ####### firstly optimize pred_cam #######
        with torch.enable_grad():
            pred_cam = pred_cam.clone().detach().requires_grad_(True)
            def closure_trans():
                # optimizer_trans.zero_grad()
                # theta = torch.cat([orient_6d, pose_6d], dim=-1).reshape(-1, 6)
                curr_pose_mtx = rotation_6d_to_matrix(pose_6d)
                curr_pose = matrix_to_axis_angle(curr_pose_mtx.view(-1, 3, 3)).view(-1, 72)

                # convert the camera parameters from the crop camera to the full camera
                img_h, img_w = img_info['img_h'], img_info['img_w']
                focal_length = img_info['focal_length']
                center = img_info['center']
                scale = img_info['scale']

                full_img_shape = torch.stack((img_h, img_w), dim=-1)
                pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)
            
                _, pred_joints = self.smpl(shape, curr_pose, pred_trans) # global joints
                verts, pred_joints_halpe = self.smpl(shape, curr_pose, pred_trans, halpe=True)

                foot_slipping_loss, foot_slipping_loss_mean = self.get_foot_slipping_loss(pred_joints, data, img_info, debug=debug, t=t, i=i)
                # _, attraction_loss, repulsion_loss, interaction_loss = self.get_interaction_loss(shape, pred_joints, pred_joints_halpe, data, return_interaction_loss=False, debug=debug)
                joint_loss_mean = loss_fn(pred_joints_halpe, target_g_joints).mean()

                foot_slipping_loss_mean = foot_slipping_loss_mean*10.0  
                # repulsion_loss = repulsion_loss.mean()*1000.0
                # attraction_loss = attraction_loss.mean()*100.0
                joint_loss_mean = joint_loss_mean*100.0

                print("foot_slipping_loss:", foot_slipping_loss_mean)
                # print("repulsion_loss:", repulsion_loss)
                # print("attraction_loss:", attraction_loss)
                print("joint_loss:", joint_loss_mean)
                all_loss = joint_loss_mean
                            # repulsion_loss + \
                            # attraction_loss + \
                            
                self.use_trans_prev = False
                if all_loss == all_loss:
                    print("all loss:", all_loss)
                    all_loss = torch.clamp(all_loss, 0, 1e6)
                    all_loss.backward()
                    print("pred_cam.grad.mean():", pred_cam.grad.mean())
                    self.trans_prev = pred_cam.clone()
                else:
                    self.use_trans_prev = True

                return all_loss
            
            # optimizer_trans = torch.optim.LBFGS([pred_cam], lr=100000, max_iter=20, line_search_fn='strong_wolfe')
            optimizer_trans = torch.optim.Adam([pred_cam], lr=0.0001, weight_decay=0.00001)
            iterations = 20
            
            for iter in range(iterations):
                closure_trans.iter = iter
                print("iter:", iter)
                optimizer_trans.step(closure_trans)

        if self.use_trans_prev:
            pred_cam = self.trans_prev

        ####### then optimize pose #######
        # contact = data['contact_correspondences']
        # affordance = torch.where(contact > 0.5, torch.ones_like(contact), torch.zeros_like(contact))

        with torch.enable_grad():
            pose_6d = pose_6d.clone().detach().requires_grad_(True)
            def closure_pose():
                # optimizer_pose.zero_grad()
                # Convert 6D rotation to axis angle
                # theta = torch.cat([orient_6d, pose_6d], dim=-1).reshape(-1, 6)
                curr_pose_mtx = rotation_6d_to_matrix(pose_6d)
                curr_pose = matrix_to_axis_angle(curr_pose_mtx.view(-1, 3, 3)).view(-1, 72)

                # convert the camera parameters from the crop camera to the full camera
                img_h, img_w = img_info['img_h'], img_info['img_w']
                focal_length = img_info['focal_length']
                center = img_info['center']
                scale = img_info['scale']

                full_img_shape = torch.stack((img_h, img_w), dim=-1)
                pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)
            
                # _, pred_joints = self.smpl(shape, curr_pose, pred_trans) # global joints
                verts, pred_joints_halpe = self.smpl(shape, curr_pose, pred_trans, halpe=True)

                # target_joint_aff = self.get_target_joints_from_affordance(affordance, pred_joints_halpe, pred_trans, data).reshape_as(pred_joints_halpe)
                # visualization
                # for i in range(batch_size*frame_length):
                #     path = data['imgname'][i]
                #     seq = path.split('/')[-3]
                #     cam = path.split('/')[-2]
                #     frame_id = path.split('/')[-1].split('.')[0]
                #     frame_id = int(frame_id)
                #     if seq == 'dance27' and cam == "Camera04" and frame_id > 60:
                #         img = cv2.imread(path)
                #         img_h, img_w = img.shape[:2]
                #         renderer = Renderer(focal_length=focal_length[0], center=(img_w/2, img_h/2), img_w=img.shape[1], img_h=img.shape[0], faces=self.smpl.faces, same_mesh_color=True)
                #         test_joints = target_joint_aff.reshape([-1,2,26,3])[i].detach().cpu().numpy()
                #         joints_front_view = renderer.render_joints_front_view(test_joints, img.copy())
                #         joints_side_view = renderer.render_joints_side_view(test_joints)
                #         joints_top_view = renderer.render_joints_top_view(test_joints)
                #         img_target_joints = np.concatenate((joints_front_view, joints_side_view, joints_top_view), axis=1)
                #         test_joints = pred_joints_halpe.reshape([-1,2,26,3])[i].detach().cpu().numpy()
                #         joints_front_view = renderer.render_joints_front_view(test_joints, img.copy())
                #         joints_side_view = renderer.render_joints_side_view(test_joints)
                #         joints_top_view = renderer.render_joints_top_view(test_joints)
                #         img_pred_joints = np.concatenate((joints_front_view, joints_side_view, joints_top_view), axis=1)
                #         img_joints = np.concatenate((img_target_joints, img_pred_joints), axis=0)
                #         cv2.imwrite('joints.jpg', img_joints)
                #         renderer.delete()

                joint_loss_mean = loss_fn(pred_joints_halpe, target_g_joints).mean()
                joint_loss_mean = joint_loss_mean*1000.0
                print("joint_loss:", joint_loss_mean)

                # verts = verts.reshape(-1, frame_length, agent_num, 6890, 3)
                # verts_a = verts[:,:,0].contiguous()
                # verts_b = verts[:,:,1].contiguous()
                
                # contact_prior_loss = self.cal_contact_prior_loss(verts_a, verts_b)
                # _, attraction_loss, repulsion_loss, interaction_loss = self.get_interaction_loss(shape, pred_joints, pred_joints_halpe, data, return_interaction_loss=False, debug=debug)
                # # interpolation_loss = self.get_interpolate_frame_penetration_loss(shape, pred_joints, pred_joints_halpe, data, img_info, interpolate_frame = 2, debug=debug,  i=closure.iter)
                # _, projection_loss = self.get_projection_loss(pred_joints_halpe, data, img_info, debug=debug)
        
                # pose_smooth_loss = 0
                # trans_smooth_loss = 0
                # if frame_length > 1:
                    # # Compute acceleration (second derivative) of pose parameters
                    # pose_acc = pose_6d[:,2:] - 2 * pose_6d[:,1:-1] + pose_6d[:,:-2]
                    # pose_smooth_loss = torch.sum(pose_acc ** 2)
                    
                    # # Optional: Add velocity (first derivative) constraint
                    # pose_vel = pose_6d[:,1:] - pose_6d[:,:-1]
                    # pose_smooth_loss += 0.5 * torch.sum(pose_vel ** 2)
                    
                #     # Compute acceleration (second derivative) of translation parameters
                #     trans_acc = pred_trans[:,2:] - 2 * pred_trans[:,1:-1] + pred_trans[:,:-2]
                #     trans_smooth_loss = torch.sum(trans_acc ** 2)
                    
                #     # Optional: Add velocity (first derivative) constraint
                #     trans_vel = pred_trans[:,1:] - pred_trans[:,:-1]
                #     trans_smooth_loss += 0.5 * torch.sum(trans_vel ** 2)
                
                # repulsion_loss = repulsion_loss.mean()*0.0
                # attraction_loss = attraction_loss.mean()*10.0
                # trans_smooth_loss = trans_smooth_loss*0.0
                # pose_smooth_loss = pose_smooth_loss*1e-3
                # projection_loss = projection_loss * 0.1
                # contact_prior_loss = contact_prior_loss*5e-3
                # print("repulsion_loss:", repulsion_loss)
                # print("attraction_loss:", attraction_loss)
                # print("projection_loss:", projection_loss)
                # print("pose_smooth_loss:", pose_smooth_loss)
                # print("trans_smooth_loss:", trans_smooth_loss)
                # print("contact_prior_loss:", contact_prior_loss)
                    
                all_loss = joint_loss_mean # + contact_prior_loss
                                # repulsion_loss + \
                #             attraction_loss + \
                #             projection_loss + \
                #             trans_smooth_loss + \
                #             pose_smooth_loss + \
                #             contact_prior_loss
                self.use_pose_6d_prev = False
                if all_loss == all_loss: # check if nan
                    print("all loss:", all_loss)
                    all_loss.backward()
                    print("pose_6d.grad.mean():", pose_6d.grad.mean())
                    # set x_t_prev to x_t
                    self.pose_6d_prev = pose_6d.clone()
                else:
                    all_loss = torch.tensor(0., dtype=torch.float32, device=pose_6d.device)
                    self.use_pose_6d_prev = True
                return all_loss

            # optimizer_pose = optim.LBFGS([pose_6d], lr=0.1, max_iter=20, line_search_fn='strong_wolfe')
            optimizer_pose = torch.optim.Adam([pose_6d], lr=0.0001, weight_decay=0.00001)
            iterations = 30
            
            for iter in range(iterations):
                print("iter:", iter)
                # closure.iter = iter
                optimizer_pose.step(closure_pose)

        if self.use_pose_6d_prev:
            pose_6d = self.pose_6d_prev

        # ####### finally optimize hand and arm #######
        # hand_arm_id = [16,18,20,22,17,19,21,23]
        # # theta = torch.cat([orient_6d, pose_6d], dim=-1).reshape(-1, 6)
        # hand_arm_pose = pose_6d.reshape(-1, 24, 6)[:, hand_arm_id, :]

        # contact = data['contact_correspondences']
        # affordance = torch.where(contact > 0.5, torch.ones_like(contact), torch.zeros_like(contact))

        # with torch.enable_grad():
        #     hand_arm_pose = hand_arm_pose.clone().detach().requires_grad_(True)
        #     def closure_hand_arm_pose():
        #         # optimizer_hand_arm_pose.zero_grad()
        #         # Convert 6D rotation to axis angle
        #         # theta = torch.cat([orient_6d, pose_6d], dim=-1).reshape(-1, 6)
        #         curr_pose_mtx = rotation_6d_to_matrix(pose_6d)
        #         curr_pose = matrix_to_axis_angle(curr_pose_mtx.view(-1, 3, 3)).view(-1, 72)

        #         # convert the camera parameters from the crop camera to the full camera
        #         img_h, img_w = img_info['img_h'], img_info['img_w']
        #         focal_length = img_info['focal_length']
        #         center = img_info['center']
        #         scale = img_info['scale']

        #         full_img_shape = torch.stack((img_h, img_w), dim=-1)
        #         pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)
            
        #         _, pred_joints = self.smpl(shape, curr_pose, pred_trans) # global joints
        #         verts, pred_joints_halpe = self.smpl(shape, curr_pose, pred_trans, halpe=True)
        #         target_joint = self.get_target_joints(affordance, pred_joints_halpe, pred_trans, data).reshape_as(pred_joints_halpe)

        #         # visualization
        #         # for i in range(batch_size*frame_length):
        #         #     path = data['imgname'][i]
        #         #     seq = path.split('/')[-3]
        #         #     cam = path.split('/')[-2]
        #         #     frame_id = path.split('/')[-1].split('.')[0]
        #         #     frame_id = int(frame_id)
        #         #     if seq == 'dance27' and cam == "Camera04" and frame_id > 60:
        #         #         img = cv2.imread(path)
        #         #         img_h, img_w = img.shape[:2]
        #         #         renderer = Renderer(focal_length=focal_length[0], center=(img_w/2, img_h/2), img_w=img.shape[1], img_h=img.shape[0], faces=self.smpl.faces, same_mesh_color=True)
        #         #         test_joints = target_joint.reshape([-1,2,26,3])[i].detach().cpu().numpy()
        #         #         joints_front_view = renderer.render_joints_front_view(test_joints, img.copy())
        #         #         joints_side_view = renderer.render_joints_side_view(test_joints)
        #         #         joints_top_view = renderer.render_joints_top_view(test_joints)
        #         #         img_target_joints = np.concatenate((joints_front_view, joints_side_view, joints_top_view), axis=1)
        #         #         test_joints = pred_joints_halpe.reshape([-1,2,26,3])[i].detach().cpu().numpy()
        #         #         joints_front_view = renderer.render_joints_front_view(test_joints, img.copy())
        #         #         joints_side_view = renderer.render_joints_side_view(test_joints)
        #         #         joints_top_view = renderer.render_joints_top_view(test_joints)
        #         #         img_pred_joints = np.concatenate((joints_front_view, joints_side_view, joints_top_view), axis=1)
        #         #         img_joints = np.concatenate((img_target_joints, img_pred_joints), axis=0)
        #         #         cv2.imwrite('joints.jpg', img_joints)
        #         #         renderer.delete()
        #         loss_fn = torch.nn.MSELoss(reduction='none') # return square error
        #         joint_loss = loss_fn(pred_joints_halpe, target_joint)
        #         joint_loss = joint_loss.mean()*1000.0
        #         print("joint_loss:", joint_loss)

        #         # verts = verts.reshape(-1, frame_length, agent_num, 6890, 3)
        #         # verts_a = verts[:,:,0].contiguous()
        #         # verts_b = verts[:,:,1].contiguous()
                
        #         # contact_prior_loss = self.cal_contact_prior_loss(verts_a, verts_b)
        #         # _, attraction_loss, repulsion_loss, interaction_loss = self.get_interaction_loss(shape, pred_joints, pred_joints_halpe, data, return_interaction_loss=False, debug=debug)
        #         # # interpolation_loss = self.get_interpolate_frame_penetration_loss(shape, pred_joints, pred_joints_halpe, data, img_info, interpolate_frame = 2, debug=debug,  i=closure.iter)
        #         # _, projection_loss = self.get_projection_loss(pred_joints_halpe, data, img_info, debug=debug)
        
        #         # pose_smooth_loss = 0
        #         # trans_smooth_loss = 0
        #         # if frame_length > 1:
        #             # # Compute acceleration (second derivative) of pose parameters
        #             # pose_acc = pose_6d[:,2:] - 2 * pose_6d[:,1:-1] + pose_6d[:,:-2]
        #             # pose_smooth_loss = torch.sum(pose_acc ** 2)
                    
        #             # # Optional: Add velocity (first derivative) constraint
        #             # pose_vel = pose_6d[:,1:] - pose_6d[:,:-1]
        #             # pose_smooth_loss += 0.5 * torch.sum(pose_vel ** 2)
                    
        #         #     # Compute acceleration (second derivative) of translation parameters
        #         #     trans_acc = pred_trans[:,2:] - 2 * pred_trans[:,1:-1] + pred_trans[:,:-2]
        #         #     trans_smooth_loss = torch.sum(trans_acc ** 2)
                    
        #         #     # Optional: Add velocity (first derivative) constraint
        #         #     trans_vel = pred_trans[:,1:] - pred_trans[:,:-1]
        #         #     trans_smooth_loss += 0.5 * torch.sum(trans_vel ** 2)
                
        #         # repulsion_loss = repulsion_loss.mean()*1000.0
        #         # attraction_loss = attraction_loss.mean()*10.0
        #         # trans_smooth_loss = trans_smooth_loss*0.0
        #         # pose_smooth_loss = pose_smooth_loss*1e-3
        #         # projection_loss = projection_loss * 0.1
        #         # contact_prior_loss = contact_prior_loss*1e-1
        #         # print("repulsion_loss:", repulsion_loss)
        #         # print("attraction_loss:", attraction_loss)
        #         # print("projection_loss:", projection_loss)
        #         # print("pose_smooth_loss:", pose_smooth_loss)
        #         # print("trans_smooth_loss:", trans_smooth_loss)
        #         # print("contact_prior_loss:", contact_prior_loss)
                    
        #         all_loss = joint_loss
        #         #             repulsion_loss
        #         #             attraction_loss + \
        #         #             projection_loss + \
        #         #             trans_smooth_loss + \
        #         #             pose_smooth_loss + \
        #         #             contact_prior_loss
        #         self.use_hand_arm_prev = False
        #         if all_loss == all_loss: # check if nan
        #             print("all loss:", all_loss)
        #             all_loss.backward()
        #             print("hand_arm_pose.grad.mean():", hand_arm_pose.grad.mean())
        #             # set x_t_prev to x_t
        #             self.hand_arm_pose_prev = hand_arm_pose.clone()
        #         else:
        #             self.use_hand_arm_prev = True
                

        #         return all_loss

        #     # optimizer_hand_arm_pose = optim.LBFGS([hand_arm_pose], lr=1, max_iter=20, line_search_fn='strong_wolfe')
        #     optimizer_hand_arm_pose = torch.optim.Adam([hand_arm_pose], lr=0.0001, weight_decay=0.00001)
        #     iterations = 20
            
        #     for iter in range(iterations):
        #         print("iter:", iter)
        #         # closure.iter = iter
        #         optimizer_hand_arm_pose.step(closure_hand_arm_pose)

        # if self.use_hand_arm_prev:
        #     hand_arm_pose = self.hand_arm_pose_prev

        shape = shape.reshape(batch_size, frame_length, agent_num, 10)
        pred_cam = pred_cam.reshape(batch_size, frame_length, agent_num, 3)
        # theta = torch.cat([orient_6d, pose_6d], dim=-1).reshape(-1, 24, 6)
        # theta[:, hand_arm_id] = hand_arm_pose
        pose_6d = pose_6d.reshape(batch_size, frame_length, agent_num, 24*6)  
        x_t = torch.cat([pose_6d, shape, pred_cam], dim=-1).reshape(batch_size, frame_length, agent_num, 157)
        return x_t
        
    def classifier_guidance_mean(self, model_output, mean, data, img_info, t=0, i=0, debug=False):
        """
        args:
            x_t: [bs, seq_len, num_person, 154]
            data: dict
            img_info: dict
        return:
            x_t: [bs, seq_len, num_person, 154] x_t under guidance if the new loss is smaller
        """
        # print("==========classifier_guidance==========")
        batch_size, frame_length, agent_num = data['features'].shape[:3]
        num_valid = batch_size * frame_length * agent_num

        with torch.enable_grad():
            mean = mean.clone().detach().requires_grad_(True)
            def closure():
                x_t = model_output + mean
                pose_6d = x_t[:,:,:,:144].reshape(-1, 6)
                shape = x_t[:,:,:,144:154].reshape(-1, 10)
                pred_cam = x_t[:,:,:,154:].reshape(-1, 3)
                # Convert 6D rotation to axis angle
                curr_pose_mtx = rotation_6d_to_matrix(pose_6d)
                curr_pose = matrix_to_axis_angle(curr_pose_mtx.view(-1, 3, 3)).view(-1, 72)

                # convert the camera parameters from the crop camera to the full camera
                img_h, img_w = img_info['img_h'], img_info['img_w']
                focal_length = img_info['focal_length']
                center = img_info['center']
                scale = img_info['scale']

                if self.use_cfg_sampler and not self.training:
                    img_h = img_h[:num_valid]
                    img_w = img_w[:num_valid]
                    focal_length = focal_length[:num_valid]
                    center = center[:num_valid]
                    scale = scale[:num_valid]
                    
                full_img_shape = torch.stack((img_h, img_w), dim=-1)
                pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)
            
                target_pose = data['pose_6d']
                target_shape = data['betas']
                target_trans = data['gt_cam_t']

                loss_fn = torch.nn.MSELoss(reduction='none')
                loss_pose = loss_fn(pose_6d, target_pose)
                loss_shape = loss_fn(shape, target_shape)
                loss_trans = loss_fn(pred_trans, target_trans)

                all_loss = loss_pose.mean() + loss_shape.mean() + loss_trans.mean()
                all_loss.backward()

                # verts, pred_joints = self.smpl(shape, curr_pose, pred_trans) # global joints
                # _ , pred_joints_halpe = self.smpl(shape, curr_pose, pred_trans, halpe=True)
                
                # lbfgs.zero_grad()
                # verts = verts.reshape(-1, frame_length, agent_num, 6890, 3)
                # verts_a = verts[:,:,0]
                # verts_b = verts[:,:,1]
                
                # contact_prior_loss = self.cal_contact_prior_loss(verts_a, verts_b)
                # _, attraction_loss, repulsion_loss, interaction_loss = self.get_interaction_loss(shape, pred_joints, pred_joints_halpe, data, return_interaction_loss=False, debug=debug, t=t, i=closure.iter)
                # interpolation_loss = self.get_interpolate_frame_penetration_loss(shape, pred_joints, pred_joints_halpe, data, img_info, interpolate_frame = 2, debug=debug, t=t, i=closure.iter)
                # _, projection_loss = self.get_projection_loss(pred_joints_halpe, data, img_info, debug=debug, t=t, i=closure.iter)
        
                # pose_smooth_loss = 0
                # trans_smooth_loss = 0
                # if frame_length > 1:
                #     # Compute acceleration (second derivative) of pose parameters
                #     pose_acc = pose_6d[:,2:] - 2 * pose_6d[:,1:-1] + pose_6d[:,:-2]
                #     pose_smooth_loss = torch.sum(pose_acc ** 2)
                    
                #     # Optional: Add velocity (first derivative) constraint
                #     pose_vel = pose_6d[:,1:] - pose_6d[:,:-1]
                #     pose_smooth_loss += 0.5 * torch.sum(pose_vel ** 2)
                    
                #     # Compute acceleration (second derivative) of translation parameters
                #     trans_acc = pred_trans[:,2:] - 2 * pred_trans[:,1:-1] + pred_trans[:,:-2]
                #     trans_smooth_loss = torch.sum(trans_acc ** 2)
                    
                #     # Optional: Add velocity (first derivative) constraint
                #     trans_vel = pred_trans[:,1:] - pred_trans[:,:-1]
                #     trans_smooth_loss += 0.5 * torch.sum(trans_vel ** 2)
                
                # repulsion_loss = repulsion_loss.mean()*10
                # attraction_loss = attraction_loss.mean()*0.0
                # trans_smooth_loss = trans_smooth_loss*0.0
                # pose_smooth_loss = pose_smooth_loss*0.0
                # interpolation_loss = interpolation_loss*0.0
                # projection_loss = projection_loss * 0.0
                # contact_prior_loss = contact_prior_loss * 50
                # print("repulsion_loss:", repulsion_loss)
                # print("attraction_loss:", attraction_loss)
                # print("interpolation_loss:", interpolation_loss)
                # print("projection_loss:", projection_loss)
                # print("pose_smooth_loss:", pose_smooth_loss)
                # print("trans_smooth_loss:", trans_smooth_loss)
                # print("contact_prior_loss:", contact_prior_loss)
                    
                # all_loss = repulsion_loss + \
                #             attraction_loss + \
                #             projection_loss + \
                #             trans_smooth_loss + \
                #             pose_smooth_loss + \
                #             interpolation_loss + \
                #             contact_prior_loss
                # print("all loss:", all_loss)
                
                # all_loss = torch.clamp(all_loss, 0, 1e6)
                # all_loss.backward(retain_graph=True)
                
                # # Apply different scaling factors to gradients
                # with torch.no_grad():
                #     mean.grad[:,:,:,:6] *= 0.0  # Scale gradients for pose_6d
                # #     mean.grad[:,:,:,6:144] *= 1.0  # Scale gradients for pose_6d
                #     mean.grad[:,:,:,144:154] *= 0.00  # Scale gradients for shape
                #     mean.grad[:,:,:,154:] *= 0.0  # Scale gradients for pred_cam

                return all_loss

            lbfgs = optim.LBFGS([mean], line_search_fn='strong_wolfe', lr=0.1)
            # lbfgs = optim.Adam([mean], lr=0.01, weight_decay=0.01)
            iterations = 1
            
            for iter in range(iterations):
                print("iter:", iter)
                closure.iter = iter
                lbfgs.step(closure)
            
        return mean
    
    def classifier_guidance_cond(self, img, t, mean, cond, img_info, data, debug = False, **kwargs):
        """
        return:
         pred: dict
         cond: [bs, seq_len, num_person, 256]
        """
        map_tensor = torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
        new_ts = map_tensor[t]
        batch_size, frame_length, agent_num = data['features'].shape[:3]
        with torch.enable_grad():
            cond = cond.clone().detach().requires_grad_(True)
            def closure():
                pred = self.inference(img, new_ts, cond, img_info, data, mean, **kwargs)
                pose_6d = pred['pred_pose6d'].reshape(-1, 6)
                shape = pred['pred_shape'].reshape(-1, 10)
                pred_cam = pred['pred_cam'].reshape(-1, 3)
                # Convert 6D rotation to axis angle
                curr_pose_mtx = rotation_6d_to_matrix(pose_6d)
                curr_pose = matrix_to_axis_angle(curr_pose_mtx.view(-1, 3, 3)).view(-1, 72)

                # convert the camera parameters from the crop camera to the full camera
                img_h, img_w = img_info['img_h'], img_info['img_w']
                focal_length = img_info['focal_length']
                center = img_info['center']
                scale = img_info['scale']

                full_img_shape = torch.stack((img_h, img_w), dim=-1)
                pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)
            
                verts, pred_joints = self.smpl(shape, curr_pose, pred_trans) # global joints
                _ , pred_joints_halpe = self.smpl(shape, curr_pose, pred_trans, halpe=True)
                
                if iter == 0:
                    lbfgs.zero_grad() # clear the gradient in the first iteration
                verts = verts.reshape(-1, frame_length, agent_num, 6890, 3)
                verts_a = verts[:,:,0]
                verts_b = verts[:,:,1]
                
                contact_prior_loss = self.cal_contact_prior_loss(verts_a, verts_b)
                _, attraction_loss, repulsion_loss, interaction_loss = self.get_interaction_loss(shape, pred_joints, pred_joints_halpe, data, return_interaction_loss=False, debug=debug, t=t, i=closure.iter)
                interpolation_loss = self.get_interpolate_frame_penetration_loss(shape, pred_joints, pred_joints_halpe, data, img_info, interpolate_frame = 2, debug=debug, t=t, i=closure.iter)
                _, projection_loss = self.get_projection_loss(pred_joints_halpe, data, img_info, debug=debug, t=t, i=closure.iter)
        
                pose_smooth_loss = 0
                trans_smooth_loss = 0
                if frame_length > 1:
                    # Compute acceleration (second derivative) of pose parameters
                    pose_acc = pose_6d[:,2:] - 2 * pose_6d[:,1:-1] + pose_6d[:,:-2]
                    pose_smooth_loss = torch.sum(pose_acc ** 2)
                    
                    # Optional: Add velocity (first derivative) constraint
                    pose_vel = pose_6d[:,1:] - pose_6d[:,:-1]
                    pose_smooth_loss += 0.5 * torch.sum(pose_vel ** 2)
                    
                    # Compute acceleration (second derivative) of translation parameters
                    trans_acc = pred_trans[:,2:] - 2 * pred_trans[:,1:-1] + pred_trans[:,:-2]
                    trans_smooth_loss = torch.sum(trans_acc ** 2)
                    
                    # Optional: Add velocity (first derivative) constraint
                    trans_vel = pred_trans[:,1:] - pred_trans[:,:-1]
                    trans_smooth_loss += 0.5 * torch.sum(trans_vel ** 2)
                
                repulsion_loss = repulsion_loss.mean()*0.0
                attraction_loss = attraction_loss.mean()*0.0
                interaction_loss = interaction_loss*1.0
                trans_smooth_loss = trans_smooth_loss*0.0
                pose_smooth_loss = pose_smooth_loss*0.0
                interpolation_loss = interpolation_loss*1.0
                projection_loss = projection_loss*0.0
                contact_prior_loss = contact_prior_loss*0.0
                # print("repulsion_loss:", repulsion_loss.mean())
                # print("attraction_loss:", attraction_loss.mean())
                print("interaction_loss:", interaction_loss)
                print("interpolation_loss:", interpolation_loss)
                # print("projection_loss:", projection_loss)
                # print("pose_smooth_loss:", pose_smooth_loss)
                # print("trans_smooth_loss:", trans_smooth_loss)
                # print("contact_prior_loss:", contact_prior_loss)
                    
                all_loss = repulsion_loss + \
                            attraction_loss + \
                            interaction_loss + \
                            projection_loss + \
                            trans_smooth_loss + \
                            pose_smooth_loss + \
                            interpolation_loss + \
                            contact_prior_loss
                    
                print("all loss:", all_loss)
                
                # clamp the loss to avoid NaN
                # all_loss = torch.clamp(all_loss, 0, 4000)
                all_loss.backward()
                
                # clip gradients to avoid NaN
                # check the gradient for each parameter
                # print("cond.grad:", cond.grad)
                # for param in [cond]:
                #     if param.grad is not None:
                #         param.grad.data.clamp_(-0.1, 0.1)
                # torch.nn.CloseInt.utils.clip_grad_norm_([cond], 0.01)
                                                
                # Apply different scaling factors to gradients
                # with torch.no_grad():
                #     x_t.grad[:,:,:,:6] *= 0.01  # Scale gradients for pose_6d
                #     x_t.grad[:,:,:,6:144] *= 1.0  # Scale gradients for pose_6d
                #     x_t.grad[:,:,:,144:154] *= 0.001  # Scale gradients for shape
                #     x_t.grad[:,:,:,154:] *= 1.0  # Scale gradients for pred_cam

                return all_loss
            
            # lbfgs = optim.LBFGS([cond], line_search_fn='strong_wolfe', lr=1, tolerance_grad=1e-8, tolerance_change=1e-11)
            lbfgs = optim.Adam([cond], lr=0.001, weight_decay=0.01)
            iterations = 10
            
            for iter in range(iterations):
                # print("iter:", iter)
                closure.iter = iter
                lbfgs.step(closure)

        return cond

    def ddim_sample_loop(self, noise, mean, cond, img_info, data, eta=0.0, **kwargs):
        indices = list(range(self.num_timesteps_test))[::-1]
        
        # # temp
        # indices = range(1)

        img = noise
        # preds = []
        for i in indices:
            # print("i: ",i) # from 4 to 0
            t = torch.tensor([i] * noise.shape[0], device=noise.device)
            # print("t: ",t)
            # i:  3
            # t:  tensor([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            # 3, 3, 3, 3, 3, 3, 3, 3], device='cuda:0')
            
            # ddim_sample
            # model_mean_type=ModelMeanType.START_X,
            # model_var_type=ModelVarType.FIXED_SMALL,
            if self.use_classifier_guidance_cond:
                pred_shape = self.ddim_sample(img, t, mean, cond, img_info, data, **kwargs)['pred_shape']
                cond = self.classifier_guidance_cond(img, t, mean, cond, img_info, data, **kwargs)
                pred = self.ddim_sample(img, t, mean, cond, img_info, data, **kwargs)
            else:
                pred = self.ddim_sample(img, t, mean, cond, img_info, data, **kwargs)
            # preds.append(pred)

            # prepare data for classifier_guidance
            data['pred_joints'] = pred['pred_joints']
            data['pred_cam_t'] = pred['pred_cam_t']

            # construct x_{t-1}
            pred_pose6d = pred['pred_pose6d']
            pred_shape = pred['pred_shape']
            pred_cam = pred['pred_cam']

            # Visualize each diffusion step
            viz_denoising = False
            if viz_denoising:
                self.visualize(pred_pose6d, pred_shape, pred_cam, data, img_info, i, name='denoising')

            model_output = torch.cat([pred_pose6d, pred_shape, pred_cam], dim=-1)
            model_output = model_output.reshape(*img.shape) # [bs, seq_len, num_person, 157]
            
            # if self.use_classifier_guidance:
            #     model_output = self.classifier_guidance(model_output, data, img_info, t, i, debug=False)
            
            model_output = model_output - mean
            
            if self.use_classifier_guidance_mean:
                # test2 apply classifier guidance on mean
                mean = self.classifier_guidance_mean(model_output, mean, data, img_info, t=0, i=0, debug=False)


            model_variance, model_log_variance = (
                    self.test_posterior_variance,
                    self.test_posterior_log_variance_clipped,
                )
            
            model_variance = extract_into_tensor(model_variance, t, img.shape)
            model_log_variance = extract_into_tensor(model_log_variance, t, img.shape)

            # model_mean_type == ModelMeanType.START_X and no need for process_xstart
            pred_xstart = model_output
            
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=img, t=t
            )

            assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == img.shape
            )
            # end of p_mean_variance

            # cond_fn = self.keypoint_guidance
            # pred_xstart = self.condition_score(cond_fn, pred_xstart, img, t, cond, img_info, data, mean)
                    
            # Usually our model outputs epsilon, but we re-derive it
            # in case we used x_start or x_prev prediction.
            eps = self._predict_eps_from_xstart(img, t, pred_xstart)

            alpha_bar = extract_into_tensor(self.test_alphas_cumprod, t, img.shape)
            alpha_bar_prev = extract_into_tensor(self.test_alphas_cumprod_prev, t, img.shape)
            sigma = (
                    eta
                    * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                    * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
            )
            # Equation 12.
            noise, temp_mean = self.generate_noise(pred_pose6d.reshape_as(data['init_pose_6d']))
            mean_pred = (
                    pred_xstart * torch.sqrt(alpha_bar_prev)
                    + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
            )
            
            # if self.use_classifier_guidance:
            #     x = mean_pred + mean
            #     x = self.classifier_guidance(x, data, img_info, t, i, debug=False)
            #     mean_pred = x - mean
                
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
            )  # no noise when t == 0
            sample = mean_pred + nonzero_mask * sigma * noise

            img = sample

            # visualization
            viz_sampling = False
            if viz_sampling:
                x_t = img + mean

                pose = x_t[:,:,:,:144].reshape(-1, 144).contiguous()
                shape = x_t[:,:,:,144:154].reshape(-1, 10).contiguous()
                pred_cam = x_t[:,:,:,154:].reshape(-1, 3).contiguous()

                self.visualize(pose, shape, pred_cam, data, img_info, i, name='grad_before')

            # visualization
            viz_sampling = False
            if viz_sampling:
                x_t = img + mean

                pose = x_t[:,:,:,:144].reshape(-1, 144).contiguous()
                shape = x_t[:,:,:,144:154].reshape(-1, 10).contiguous()
                pred_cam = x_t[:,:,:,154:].reshape(-1, 3).contiguous()

                self.visualize(pose, shape, pred_cam, data, img_info, i, name='grad_after')

        return pred

    def ddim_sample(self, x, ts, mean, cond, img_info, data, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]

        # if self.use_phys or self.use_proj_grad:
        #     if new_ts[0] < self.num_timesteps * 0.2:
        #         pen_loss, proj_grads = self.eval_physical_plausibility(new_ts, x, mean, img_info)
        #         proj_grads = img_info['pred_keypoints'].reshape(-1, 78)
        #     else:
        #         proj_grads = torch.zeros((x.shape[0], x.shape[1], x.shape[2], 26*3), dtype=x.dtype, device=x.device)
        #         pen_loss = -1 * torch.ones((x.shape[0], x.shape[1], 2), dtype=x.dtype, device=x.device)
        #         pen_loss = pen_loss.reshape(-1, 1)
        #         proj_grads = proj_grads.reshape(-1, 26*3)

        # if self.use_phys:
        #     cond = torch.cat([cond, pen_loss], dim=1)
        # if self.use_proj_grad:
        #     cond = torch.cat([cond, proj_grads], dim=1)

        pred = self.inference(x, new_ts, cond, img_info, data, mean, **kwargs)

        return pred
    
    def inference(self, x_t, t, cond, img_info, data, mean, **kwargs):
        # x_t = torch.ones_like(x_t) * 10.0
        # try to get control net from the kwargs
        if self.use_cfg_sampler and not self.training:
            x_t = torch.cat([x_t, x_t], dim=0)
            t = torch.cat([t, t], dim=0)
            
        if 'control_net' in kwargs:
            control_net = kwargs['control_net']
        else:
            control_net = None
            
            
        batch_size, frame_length, agent_num = data['features'].shape[:3]
        num_valid = batch_size * frame_length * agent_num

        mean = mean.reshape(-1, self.input_feats)

        x_a, x_b = x_t[:,:,0], x_t[:,:,1] # x_a: [bs, seq_len, 154], x_b: [bs, seq_len, 154]
        t = t[:,None, None].repeat(1, frame_length, agent_num)

        mask = None
        if mask is not None:
            mask = mask[...,0]

        emb = self.embed_timestep(t.reshape(-1)) + self.feature_embed(cond)
        emb = emb.reshape(-1, frame_length, agent_num, self.latent_dim)
        # emb_a = self.sequence_pos_encoder(emb[:,:,0])
        # emb_b = self.sequence_pos_encoder(emb[:,:,1])
        # emb = torch.cat([emb_a[:,:,None], emb_b[:,:,None]], dim=2)

        a_emb = self.motion_embed(x_a)
        b_emb = self.motion_embed(x_b) # [bs, seq_len, 154]
        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)
        
        if control_net is not None:
            h_prev = torch.cat([h_a_prev, h_b_prev], dim=2)
            control_a, control_b = control_net(h_prev, emb, img_info, data, mean)

        if mask is None:
            mask = torch.ones(batch_size, frame_length).to(x_a.device)
        key_padding_mask = ~(mask > 0.5)

        counterpart_mask = torch.ones(batch_size, frame_length, 1).to(x_a.device)
        counterpart_mask[data['single_person']>0] = 0.
        if self.use_cfg_sampler and not self.training:
            mask = torch.cat([mask, mask], dim=0)
            key_padding_mask = torch.cat([key_padding_mask, key_padding_mask], dim=0)
            counterpart_mask = torch.cat([counterpart_mask, counterpart_mask], dim=0)

        for i,block in enumerate(self.blocks):
            h_a = block(h_a_prev, h_b_prev * counterpart_mask, emb[:,:,0], key_padding_mask)
            h_b = block(h_b_prev, h_a_prev * counterpart_mask, emb[:,:,1], key_padding_mask) # (batch_size, frame_length, latent_dim)
            if control_net is not None:
                h_a = h_a + control_a[i]
                h_b = h_b + control_b[i]
            h_a_prev = h_a
            h_b_prev = h_b

        features = torch.cat([h_a[:,:,None], h_b[:,:,None]], dim=2)
        features = features.reshape(-1, self.latent_dim)
        
        print("features.shape:", features.shape)
        print("img_info['bbox_info'].shape:", img_info['bbox_info'].shape)

        xc = torch.cat([features, img_info['bbox_info']], 1)    
        
        pred_pose6d = self.head(xc).view(-1, 144)
        pred_shape = self.shape_head(xc).view(-1, 10)
        pred_cam = self.cam_head(xc).view(-1, 3)

        if self.use_cfg_sampler and not self.training:
            pred_pose6d_cond = pred_pose6d[:num_valid]
            pred_shape_cond = pred_shape[:num_valid]
            pred_cam_cond = pred_cam[:num_valid]
            pred_pose6d_uncond = pred_pose6d[num_valid:]
            pred_shape_uncond = pred_shape[num_valid:]
            pred_cam_uncond = pred_cam[num_valid:]
            pred_kpt = img_info['pred_keypoints'][:num_valid].reshape(-1, 26, 3)
            conf = pred_kpt[:,:,2].mean(dim=1)
            conf = conf * self.cfg_scale
            conf = torch.clamp(conf, 0.95, 1) # limit the confidence to be at least 0.8

            pred_pose6d = conf[:, None] * pred_pose6d_cond + (1 - conf[:, None]) * pred_pose6d_uncond
            pred_shape = conf[:, None] * pred_shape_cond + (1 - conf[:, None]) * pred_shape_uncond
            pred_cam = conf[:, None] * pred_cam_cond + (1 - conf[:, None]) * pred_cam_uncond
            
        # pred_pose6d = mean[:,:144]
        pred_pose6d = pred_pose6d + mean[:,:144]
        pred_shape = pred_shape + mean[:,144:154]
        # pred_shape = data['init_shape'].reshape(-1, 10)
        pred_cam = pred_cam + mean[:,154:157]    

        
        valid_num = data['valid'].sum() if 'valid' in data else 1
        if self.use_classifier_guidance and not self.training and t[0,0,0] < self.num_timesteps * 0.2 and valid_num > 0:
            x = torch.cat([pred_pose6d, pred_shape, pred_cam], dim=-1).reshape(-1, frame_length, agent_num, 157)
            x = self.classifier_guidance(x, data, img_info, t, 0, debug=False)
            pred_pose6d = x[...,:144].reshape(-1, 144)
            pred_shape = x[...,144:154].reshape(-1, 10)
            pred_cam = x[...,154:].reshape(-1, 3)
        
        pred_rotmat = rotation_6d_to_matrix(pred_pose6d.reshape(-1,6)).view(-1, 24, 3, 3)
        pred_pose =  matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(-1, 72)

        # convert the camera parameters from the crop camera to the full camera
        img_h, img_w = img_info['img_h'], img_info['img_w']
        focal_length = img_info['focal_length']
        center = img_info['center']
        scale = img_info['scale']
        
        if self.use_cfg_sampler and not self.training:
            img_h = img_h[:num_valid]
            img_w = img_w[:num_valid]
            focal_length = focal_length[:num_valid]
            center = center[:num_valid]
            scale = scale[:num_valid]

        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        pred_trans = cam_crop2full(pred_cam, center, scale, full_img_shape, focal_length)
        
        temp_trans = torch.zeros((pred_rotmat.shape[0], 3), dtype=pred_rotmat.dtype, device=pred_rotmat.device)

        if self.smooth and not self.training and t[0,0,0] < self.num_timesteps * 0.2:
            pred_shape = pred_shape.detach().cpu().numpy().reshape(batch_size, frame_length, agent_num, 10)
            pred_pose = pred_pose.detach().cpu().numpy().reshape(batch_size, frame_length, agent_num, 24, 3)
            pred_trans = pred_trans.detach().cpu().numpy().reshape(batch_size, frame_length, agent_num, 3)
            pred_shape = filters.gaussian_filter1d(pred_shape, 0.5, axis=1, ).reshape(-1, 10)
            pred_pose = filters.gaussian_filter1d(pred_pose, 0.5, axis=1, ).reshape(-1, 24, 3)
            pred_trans = filters.gaussian_filter1d(pred_trans, 0.5, axis=1, ).reshape(-1, 3)
            pred_shape = torch.tensor(pred_shape, dtype=temp_trans.dtype, device=temp_trans.device)
            pred_pose = torch.tensor(pred_pose, dtype=temp_trans.dtype, device=temp_trans.device)
            pred_trans = torch.tensor(pred_trans, dtype=temp_trans.dtype, device=temp_trans.device)
        
        pred_verts, pred_joints = self.smpl(pred_shape, pred_pose, temp_trans, halpe=True)

        if self.smooth and not self.training and t[0,0,0] < self.num_timesteps * 0.2:
            pred_joints = pred_joints.detach().cpu().numpy().reshape(batch_size, frame_length, agent_num, 26, 3)
            pred_joints = filters.gaussian_filter1d(pred_joints, 0.5, axis=1, ).reshape(-1, 26, 3)
            pred_joints = torch.tensor(pred_joints, dtype=temp_trans.dtype, device=temp_trans.device)
            
        camera_center = torch.stack([img_w/2, img_h/2], dim=-1)
        pred_keypoints_2d = perspective_projection(pred_joints + pred_trans[:,None,:],
                                                rotation=torch.eye(3, device=pred_pose.device).unsqueeze(0).expand(num_valid, -1, -1),
                                                translation=torch.zeros(3, device=pred_pose.device).unsqueeze(0).expand(num_valid, -1),
                                                focal_length=focal_length,
                                                camera_center=camera_center)

        pred_keypoints_2d = (pred_keypoints_2d - center[:,None,:]) / 256 #constants.IMG_RES


        pred = {'pred_pose':pred_pose, # SMPL theta
                'pred_pose6d':pred_pose6d, # SMPL theta in 6D
                'pred_shape':pred_shape, # SMPL beta
                'pred_cam_t':pred_trans, # translation in camera space
                'pred_cam':pred_cam, # camera parameters?
                'pred_rotmat':pred_rotmat, # SMPL theta in rotation matrix
                'pred_verts':pred_verts, # verts in zero trans
                'pred_joints':pred_joints, # joints in zero trans
                'focal_length':focal_length,\
                'pred_keypoints_2d':pred_keypoints_2d, # keypoints in 2D
            }

        
        output_init = True
        if output_init:
            init_pose = data['init_pose'].reshape(-1, 72)
            init_verts, init_joints = self.smpl(pred_shape, init_pose, temp_trans, halpe=True)
            pred['init_verts'] = init_verts
            pred['init_joints'] = init_joints
            
        output_comp = True
        if output_comp and not self.training and 'comp_pose_6d' in data:
            comp_pose_6d = data['comp_pose_6d'].reshape(-1, 6)
            comp_pose_mtx = rotation_6d_to_matrix(comp_pose_6d)
            comp_pose = matrix_to_axis_angle(comp_pose_mtx.view(-1, 3, 3)).view(-1, 72)
            comp_verts, comp_joints = self.smpl(pred_shape, comp_pose, temp_trans, halpe=True)
            pred['comp_verts'] = comp_verts
            pred['comp_joints'] = comp_joints
        return pred


