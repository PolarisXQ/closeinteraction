'''
 @FileName    : reconstruction_feature_data.py
 @EditTime    : 2024-04-15 15:12:01
 @Author      : Buzhen Huang
 @Email       : buzhenhuang@outlook.com
 @Description : 
'''

import os
import torch
import numpy as np
import cv2
from utils.geometry import estimate_translation_np, orient_world_to_camera_np
from utils.imutils import get_crop, keyp_crop2origin, surface_projection, img_crop2origin
from datasets.base import base
import constants
from utils.rotation_conversions import *
import random
import pickle
from utils.geometry import perspective_projection

class Reconstruction_Feature_Data(base):
    def __init__(self, train=True, dtype=torch.float32, data_folder='', name='', smpl=None, frame_length=16, sub_sequence=''):
        super(Reconstruction_Feature_Data, self).__init__(train=train, dtype=dtype, data_folder=data_folder, name=name, smpl=smpl)

        self.max_people = 2
        self.frame_length = frame_length
        self.dataset_name = name
        self.sub_sequence = sub_sequence
        self.joint_dataset = ['Panoptic', 'JTA']
        
        dataset_name = 'PW3D'

        if dataset_name == 'Hi4D':
            if self.is_train:
                # dataset_annot = os.path.join(self.dataset_dir, 'annot/train.pkl')
                dataset_annot = '/root/autodl-fs/Hi4D/train.pkl'
            else:
                # dataset_annot = os.path.join(self.dataset_dir, 'annot/test.pkl')
                dataset_annot = '/root/autodl-fs/Hi4D/test.pkl'
            
            annots = self.load_pkl(dataset_annot)
            
        elif dataset_name == 'PW3D':
            annots = {'pose2ds_pred':[], #ok 
                      'pose2ds':[], # calculated
                        'imnames':[], #ok 
                        'img_size':[], # known
                        'intris':[], # in sequence file
                        'features':[], #ok 
                        'center':[], #ok 
                        'patch_scale':[], #ok 
                        'genders':[], # in sequence file
                        'init_poses':[], #ok
                        'poses':[], # in sequence file
                        'shapes':[], # in sequence file
                        'valid':[], #ok
                        'trans':[], # in sequence file
                        }
            dataroot = '/root/autodl-fs/PW3D/4dhuman_data'
            # find pkl files in dataroot recursively
            mode = 'train' if self.is_train else 'test'
            for root, dirs, files in os.walk(dataroot): 
                for file in files:
                    if file.endswith('.pkl'):
                        # check if sequence_data.pkl exists
                        if not os.path.exists(os.path.join(dataroot.replace('4dhuman_data',f'sequenceFiles/{mode}'), file)):
                            print('sequence_data.pkl not found:', os.path.join(dataroot.replace('4dhuman_data',f'sequenceFiles/{mode}'), file))
                            continue
                        if file in ['downtown_upstairs_00.pkl', 'downtown_downstairs_00.pkl']:
                            continue
                        dataset_annot_seq = os.path.join(dataroot, file)
                        annots_seq = self.load_pkl(dataset_annot_seq) # dict_keys(['pose2ds_pred', 'imnames', 'features', 'center', 'patch_scale', 'init_poses', 'init_beta', 'init_trans'])
                        for key in annots.keys():
                            if key in annots_seq.keys():
                                annots[key].append(annots_seq[key])
                        # repeat 1080, 1920 for each image
                        if file.startswith('courtyard') or file.startswith('office'):
                            # annots_seq['img_size'] = np.array([1920//2, 1080//2]).reshape(1, 2).repeat(len(annots_seq['imnames']), axis=0)
                            annots_seq['img_size'] = np.array([1920, 1080]).reshape(1, 2).repeat(len(annots_seq['imnames']), axis=0)
                        elif file.startswith('downtown'):
                            # annots_seq['img_size'] = np.array([1080//2, 1920//2]).reshape(1, 2).repeat(len(annots_seq['imnames']), axis=0)
                            annots_seq['img_size'] = np.array([1080, 1920]).reshape(1, 2).repeat(len(annots_seq['imnames']), axis=0)
                            if file.startswith('downtown_bar') or file.startswith('downtown_bus') or file.startswith('downtown_car') or file.startswith('downtown_warmWelcome'):
                                annots_seq['img_size'] = np.array([1920, 1080]).reshape(1, 2).repeat(len(annots_seq['imnames']), axis=0)
                        else:
                            raise ValueError('Unknown sequence name: {}'.format(file))
                        annots['img_size'].append(annots_seq['img_size'])
                        # add 'imageFiles/{seq_name}/' to imnames
                        for i in range(len(annots_seq['imnames'])):
                            annots_seq['imnames'][i] = os.path.join('imageFiles', file.split('.')[0], annots_seq['imnames'][i])
                        sequence_data = self.load_pkl(os.path.join(dataroot.replace('4dhuman_data',f'sequenceFiles/{mode}'), file))
                        poses = np.array(sequence_data['poses']).transpose(1,0,2) # [frame_length, 2, 72]
                        poses2d = np.array(sequence_data['poses2d']).transpose(1,0,3,2) # [frame_length, 2, 26]
                        shapes = np.array(sequence_data['betas'])[:,:10] # [2, 10]
                        # repeat to [frame_length, 2, 10]
                        shapes = shapes.reshape(1, 2, 10).repeat(len(annots_seq['imnames']), axis=0)
                        
                        jointPositions = np.array(sequence_data['jointPositions']).transpose(1,0,2) # [frame_length, 2, 72]
                        trans = jointPositions[:,:,:3] # [frame_length, 2, 3]
                        cam_poses = np.array(sequence_data['cam_poses']) # [frame_length, 4, 4]
                        # convert trans to cam space
                        cam_rot = cam_poses[:,:3,:3]
                        cam_trans = cam_poses[:,:3,3] # [frame_length, 3]
                        trans = np.einsum('bij,bkj->bki', cam_rot, trans) # [frame_length, 2, 3]
                        trans = trans + cam_trans.reshape(-1, 1, 3)
                        orient = poses[:,:,:3]
                        orient_a = orient_world_to_camera_np(orient[:,0,:], cam_rot)
                        orient_b = orient_world_to_camera_np(orient[:,1,:], cam_rot)
                        poses[:,0,:3] = orient_a
                        poses[:,1,:3] = orient_b
                        intris = np.array(sequence_data['cam_intrinsics']) # [3, 3]
                        # repeat to [frame_length, 2, 3, 3]
                        intris = intris.reshape(1, 1, 3, 3).repeat(len(annots_seq['imnames']), axis=0).repeat(2, axis=1)
                        genders_tmp = sequence_data['genders']
                        genders = []
                        for g in genders_tmp:
                            if g == 'f':
                                genders.append(0)
                            elif g == 'm':
                                genders.append(1)
                        genders = np.array(genders)
                        # repeat to [frame_length, 2]
                        genders = genders.reshape(1, 2).repeat(len(annots_seq['imnames']), axis=0)
                        annots['poses'].append(poses)
                        annots['pose2ds'].append(poses2d)
                        annots['shapes'].append(shapes)
                        annots['trans'].append(trans)
                        annots['intris'].append(intris)
                        annots['genders'].append(genders)
                        annots['valid'].append(np.ones((len(annots_seq['imnames']), 2, 1), dtype=np.float32))

                        
        try:
            self.valid = annots['valid']
        except:
            self.valid = []

        self.pred_pose2ds = annots['pose2ds_pred']
        self.pose2ds = annots['pose2ds']
        self.imnames = annots['imnames']
        # self.bboxs = annots['bbox']
        self.img_size = annots['img_size']
        self.intris = annots['intris']
        self.features = annots['features']
        self.centers = annots['center']
        self.scales = annots['patch_scale']
        self.genders = annots['genders']
        self.init_poses = annots['init_poses']
        self.poses = annots['poses']
        self.shapes = annots['shapes']
        self.trans = annots['trans']
        # self.contact_corres = annots['contact']
        # self.contact_sub2 = annots['gt_contact_sub2']

        del annots

        self.iter_list = []
        self.seq_len = []
        self.idx = []
        for i in range(len(self.features)):
            if self.is_train:
                for n in range(0, (len(self.features[i]) - self.frame_length)):
                    self.iter_list.append([i, n])
                    self.seq_len.append(len(self.features[i]))
                    self.idx.append(n)
            else:
                if self.frame_length == 64:
                    step = self.frame_length // 4
                elif self.frame_length == 16:
                    step = self.frame_length
                else:
                    print('frame length error')
                for n in range(0, (len(self.features[i]) - self.frame_length), step):
                    self.iter_list.append([i, n])
                    self.seq_len.append(len(self.features[i]))
                    self.idx.append(n)
        self.len = len(self.iter_list)


    def vis_input(self, image, pred_keypoints, keypoints, pose, betas, trans, valids, new_shapes, new_xs, new_ys, old_xs, old_ys, focal_length, img_h, img_w):
        # Show image
        image = image.copy()
        self.vis_img('img', image)

        # Show keypoints
        for key, valid, new_shape, new_x, new_y, old_x, old_y in zip(keypoints, valids, new_shapes, new_xs, new_ys, old_xs, old_ys):
            if valid == 1:
                key = keyp_crop2origin(key.clone(), new_shape, new_x, new_y, old_x, old_y)
                # keypoints = keypoints[:,:-1].detach().numpy() * constants.IMG_RES + center.numpy()
                key = key[:,:-1].astype(np.int)
                for k in key:
                    image = cv2.circle(image, tuple(k), 3, (0,0,255), -1)
        # self.vis_img('keyp', image)

        # Show keypoints
        for key, valid, new_shape, new_x, new_y, old_x, old_y in zip(pred_keypoints, valids, new_shapes, new_xs, new_ys, old_xs, old_ys):
            if valid == 1:
                key = keyp_crop2origin(key.clone(), new_shape, new_x, new_y, old_x, old_y)
                # keypoints = keypoints[:,:-1].detach().numpy() * constants.IMG_RES + center.numpy()
                key = key[:,:-1].astype(np.int)
                for k in key:
                    image = cv2.circle(image, tuple(k), 3, (0,255,0), -1)
        self.vis_img('keyp', image)
        

        # Show SMPL
        pose = pose.reshape(-1, 72)[valids==1]
        betas = betas.reshape(-1, 10)[valids==1]
        trans = trans.reshape(-1, 3)[valids==1]
        extri = np.eye(4)
        intri = np.eye(3)
        intri[0][0] = focal_length
        intri[1][1] = focal_length
        intri[0][2] = img_w / 2
        intri[1][2] = img_h / 2
        verts, joints = self.smpl(betas, pose, trans)
        for vert in verts:
            vert = vert.detach().numpy()
            projs, image = surface_projection(vert, self.smpl.faces, extri, intri, image.copy(), viz=False)
        self.vis_img('smpl', image)

    def estimate_trans_cliff(self, joints, keypoints, focal_length, img_h, img_w):
        # print(joints.shape)
        # print(keypoints.shape)
        joints = joints.detach().numpy()
        # keypoints[:,:-1] = keypoints[:,:-1] * constants.IMG_RES + np.array(center)
        if keypoints.shape[0] == 18:
            halpe2coco = [0,18,6,8,10,5,7,9,12,14,16,11,13,15,2,1,4,3]
            joints = joints[halpe2coco]
        
        gt_cam_t = estimate_translation_np(joints, keypoints[:,:2], keypoints[:,2], focal_length=focal_length, center=[img_w/2, img_h/2])
        return gt_cam_t
    
    def copy_data(self, gt_keyps, pred_keyps, poses, init_poses, shapes, features, centers, scales, genders, focal_lengthes, valid):
        gt_keyps = np.repeat(gt_keyps, self.max_people, axis=1)
        gt_keyps[:,1] = gt_keyps[::-1,1]

        pred_keyps = np.repeat(pred_keyps, self.max_people, axis=1)
        pred_keyps[:,1] = pred_keyps[::-1,1]

        poses = np.repeat(poses, self.max_people, axis=1)
        poses[:,1] = poses[::-1,1]

        init_poses = np.repeat(init_poses, self.max_people, axis=1)
        init_poses[:,1] = init_poses[::-1,1]

        shapes = np.repeat(shapes, self.max_people, axis=1)
        shapes[:,1] = shapes[::-1,1]

        features = np.repeat(features, self.max_people, axis=1)
        features[:,1] = features[::-1,1]

        centers = np.repeat(centers, self.max_people, axis=1)
        centers[:,1] = centers[::-1,1]

        scales = np.repeat(scales, self.max_people, axis=1)
        scales[:,1] = scales[::-1,1]

        focal_lengthes = np.repeat(focal_lengthes, self.max_people, axis=1)
        focal_lengthes[:,1] = focal_lengthes[::-1,1]

        valid = np.repeat(valid, self.max_people, axis=1)
        valid[:,1] = valid[::-1,1]

        genders = np.repeat(genders, self.max_people, axis=0)

        return gt_keyps, pred_keyps, poses, init_poses, shapes, features, centers, scales, genders, focal_lengthes, valid

    def apply_random_mask(self, features, mask_prob=1.0, mask_prob_part=0.5):
        mask = np.random.rand(*features.shape) <= mask_prob # if set mask_prob=1.0, then all features will be masked
        mask_type = 'all'
        if mask_type == 'all':
            features[mask] = 0
        elif mask_type == 'part':
            mask = mask & (np.random.rand(*features.shape) < mask_prob_part)
            features[mask] = 0
        return features

    # Data preprocess
    def create_data(self, index=0):
        
        load_data = {}

        seq_ind, start    = self.iter_list[index]
        seq_len           = self.seq_len[index]

        gap = 1
        ind = [start+k*gap for k in range(self.frame_length)]
        
        num_people        = len(self.pose2ds[seq_ind][0])
        seq_length        = len(self.pose2ds[seq_ind])
        gt_keyps          = np.array(self.pose2ds[seq_ind], dtype=self.np_type).reshape(seq_length, num_people, -1, 3)[ind]
        pred_keyps        = np.array(self.pred_pose2ds[seq_ind], dtype=self.np_type).reshape(seq_length, num_people, -1, 3)[ind]
        poses             = np.array(self.poses[seq_ind], dtype=self.np_type)[ind]
        init_poses        = np.array(self.init_poses[seq_ind], dtype=self.np_type)[ind]
        shapes            = np.array(self.shapes[seq_ind], dtype=self.np_type)[ind]
        features          = np.array(self.features[seq_ind], dtype=self.np_type)[ind]
        centers           = np.array(self.centers[seq_ind], dtype=self.np_type)[ind]
        scales            = np.array(self.scales[seq_ind], dtype=self.np_type)[ind]
        img_size          = np.array(self.img_size[seq_ind], dtype=self.np_type)[ind][:,np.newaxis,:].repeat(2, axis=1)
        focal_lengthes    = np.array(self.intris[seq_ind], dtype=self.np_type)[ind][:,:,0,0]
        imgnames = [os.path.join(self.dataset_dir, path) for path in np.array(self.imnames[seq_ind])[ind].tolist()]
        genders           = np.array(self.genders[seq_ind], dtype=self.np_type)[0]
        translation             = np.array(self.trans[seq_ind], dtype=self.np_type)[ind]
        # contact_corres           = np.array(self.contact_corres[seq_ind], dtype=self.np_type)[ind]
        # contact_sub2 = np.array(self.contact_sub2[seq_ind], dtype=self.np_type)[ind]
        
        if self.dataset_name == 'PW3D':
            centers = centers * 2.
            scales = scales * 2.
        if len(self.valid) > 0:
            valid         = np.array(self.valid[seq_ind], dtype=self.np_type)[ind].reshape(self.frame_length, num_people)
        else:
            valid         = np.ones((self.frame_length, num_people), dtype=self.np_type)

        img_hs = img_size[:,:,0]
        img_ws = img_size[:,:,1]
        
        if num_people < self.max_people:
            single_person = np.ones((self.frame_length,), dtype=self.np_type)
            gt_keyps, pred_keyps, poses, init_poses, shapes, features, centers, scales, genders, focal_lengthes, valid = self.copy_data(gt_keyps, pred_keyps, poses, init_poses, shapes, features, centers, scales, genders, focal_lengthes, valid)
        else:
            single_person = np.zeros((self.frame_length,), dtype=self.np_type)

        vertss, jointss, jointss_smpl, gt_transs = [], [], [], []
        for i in range(self.max_people):
            gender = genders[i]
            if gender == 0:
                smpl_model = self.smpl_female
            elif gender == 1:
                smpl_model = self.smpl_male
            else:
                smpl_model = self.smpl

            with torch.no_grad():
                temp_pose = torch.from_numpy(poses[:,i]).reshape(-1, 72).contiguous()
                temp_shape = torch.from_numpy(shapes[:,i]).reshape(-1, 10).contiguous()
                temp_trans = torch.zeros((self.frame_length, 3), dtype=torch.float32)
                verts, joints = smpl_model(temp_shape, temp_pose, temp_trans, halpe=True)
                _, joints_smpl = smpl_model(temp_shape, temp_pose, temp_trans, halpe=False)

            temp_keyps = gt_keyps[:,i].reshape(self.frame_length, -1, 3)
            gt_trans = []
            for joint, keyps, img_h, img_w, focal_length in zip(joints, temp_keyps, img_hs[:,i], img_ws[:,i], focal_lengthes[:,i]):
                try:
                    trans = self.estimate_trans_cliff(joint, keyps, focal_length, img_h, img_w)
                except:
                    trans = np.zeros((3,), dtype=np.float32)
                gt_trans.append(trans)

            gt_trans = torch.from_numpy(np.array(gt_trans, dtype=self.np_type))[:,None]
            
            conf = torch.ones((self.frame_length, 26, 1)).float()
            joints = torch.cat([joints, conf], dim=-1)[:,None]
            verts = verts[:,None]

            gt_transs.append(gt_trans)
            jointss.append(joints)
            jointss_smpl.append(joints_smpl[:,None])
            vertss.append(verts)

        has_3d = np.ones((self.frame_length, self.max_people), dtype=self.np_type)
        has_smpls = np.ones((self.frame_length, self.max_people), dtype=self.np_type)

        vertss = torch.cat(vertss, dim=1)
        gt_joints = torch.cat(jointss, dim=1)
        gt_joints_smpl = torch.cat(jointss_smpl, dim=1)
        gt_trans = torch.cat(gt_transs, dim=1)

        pelvis = gt_joints_smpl[:, :, 0, :3]
        translation = translation - pelvis.detach().numpy()
        
        if self.dataset_name == 'PW3D':
            num_valid = self.frame_length * self.max_people
            img_w = img_ws.reshape(-1)
            img_h = img_hs.reshape(-1)
            focal_length = focal_lengthes.reshape(-1)
            camera_center = torch.stack([torch.tensor(img_w, dtype=torch.float32), torch.tensor(img_h, dtype=torch.float32)], dim=-1)
            pose2d = perspective_projection((gt_joints[..., :3] + translation[..., None, :] + pelvis[..., None, :]).reshape(num_valid, 26, 3),
                                                rotation=torch.eye(3, device=joint.device).unsqueeze(0).expand(num_valid, -1, -1),
                                                translation=torch.zeros(3, device=joint.device).unsqueeze(0).expand(num_valid, -1),
                                                focal_length=torch.tensor(focal_length),
                                                camera_center=camera_center)
            gt_keyps = pose2d.reshape(self.frame_length, self.max_people, -1, 2)
            conf = torch.ones((self.frame_length, self.max_people, 26, 1)).float()
            gt_keyps = np.concatenate((gt_keyps, conf), axis=-1)
        gt_keyps[...,:2] = (gt_keyps[...,:2] - centers[:,:,np.newaxis,:]) / 256
        keypoints = torch.from_numpy(gt_keyps).float()

        if self.dataset_name == 'PW3D':
            pred_keyps = pred_keyps * 2.
        pred_keyps[...,:2] = (pred_keyps[...,:2] - centers[:,:,np.newaxis,:]) / 256
        pred_keypoints = torch.from_numpy(pred_keyps).float()

        pose_6d = torch.from_numpy(poses).reshape(-1, 72).reshape(-1, 3)
        pose_6d = axis_angle_to_matrix(pose_6d)
        pose_6d = matrix_to_rotation_6d(pose_6d)
        pose_6d = pose_6d.reshape(self.frame_length, self.max_people, -1)

        init_pose_6d = torch.from_numpy(init_poses).reshape(-1, 72).reshape(-1, 3)
        init_pose_6d = axis_angle_to_matrix(init_pose_6d)
        init_pose_6d = matrix_to_rotation_6d(init_pose_6d)
        init_pose_6d = init_pose_6d.reshape(self.frame_length, self.max_people, -1)

        # diff_fn = torch.nn.MSELoss()
        # for i in range(self.frame_length):
        #     dis_center = np.linalg.norm(centers[i,0] - centers[i,1])
        #     # print('Distance center:', dis_center)
        #     pose_diff = diff_fn(torch.from_numpy(init_poses[i,0]), torch.from_numpy(init_poses[i,1]))
        #     # print('Pose diff:', pose_diff)
        #     # if dis_center.mean() < 5 and pose_diff < 0.1:
        #     if pose_diff < 0.1:
        #         valid[i] = np.zeros((num_people,), dtype=self.np_type)
        #         # # print('Invalid frame:', imgnames[i])
        #         # depth_a = gt_trans[i,0,2]
        #         # depth_b = gt_trans[i,1,2]
        #         # closer_agent = 0 if depth_a < depth_b else 1
        #         # # features[i,1-closer_agent,:] = self.apply_random_mask(features[i,1-closer_agent,:])

        #         # # replace init_pose with pose
        #         # init_poses[i,1-closer_agent] = poses[i,1-closer_agent]
        #         # init_pose_6d[i,1-closer_agent] = pose_6d[i,1-closer_agent]
        
        diff_fn = torch.nn.MSELoss()
        # gt_kpts_a = gt_keyps[:,0]
        # gt_kpts_b = gt_keyps[:,1]
        # pred_kpts_a = pred_keyps[:,0]
        # pred_kpts_b = pred_keyps[:,1]
        gt_pose6d_a = pose_6d[:,0]
        gt_pose6d_b = pose_6d[:,1]
        pred_pose_a = init_pose_6d[:,0]
        pred_pose_b = init_pose_6d[:,1]
        gt_rotmat_a = rotation_6d_to_matrix(gt_pose6d_a.reshape(-1,6)).view(-1, 24, 3, 3)
        gt_rotmat_b = rotation_6d_to_matrix(gt_pose6d_b.reshape(-1,6)).view(-1, 24, 3, 3)
        pred_rotmat_a = rotation_6d_to_matrix(pred_pose_a.reshape(-1,6)).view(-1, 24, 3, 3)
        pred_rotmat_b = rotation_6d_to_matrix(pred_pose_b.reshape(-1,6)).view(-1, 24, 3, 3)
        diff_pose_a = diff_fn(gt_rotmat_a, pred_rotmat_a).mean()
        diff_pose_b = diff_fn(gt_rotmat_a, pred_rotmat_b).mean()
        if diff_pose_a > diff_pose_b: # the order is wrong
            # exchange order of agent for features, init_pose, init_pose_6d, pred_keypoints, centers, scales
            features_a = features[:,0]
            features_b = features[:,1]
            features = np.stack((features_b, features_a), axis=1)
            init_pose_a = init_poses[:,0]
            init_pose_b = init_poses[:,1]
            init_poses = np.stack((init_pose_b, init_pose_a), axis=1)
            init_pose_6d_a = init_pose_6d[:,0]
            init_pose_6d_b = init_pose_6d[:,1]
            init_pose_6d = torch.stack((init_pose_6d_b, init_pose_6d_a), dim=1)
            pred_keypoints_a = pred_keypoints[:,0]
            pred_keypoints_b = pred_keypoints[:,1]
            pred_keypoints = torch.stack((pred_keypoints_b, pred_keypoints_a), dim=1)
            centers_a = centers[:,0]
            centers_b = centers[:,1]
            centers = np.stack((centers_b, centers_a), axis=1)
            scales_a = scales[:,0]
            scales_b = scales[:,1]
            scales = np.stack((scales_b, scales_a), axis=1)
            
                
        seq_name = imgnames[0].split('/')[-2]
        for f in range(self.frame_length):
            frame_num = imgnames[f].split('/')[-1].split('.')[-2].split('_')[-1]
            if int(frame_num) > 277 and int(frame_num) < 452 and seq_name == 'courtyard_rangeOfMotions_01':
                valid[f] = np.zeros((num_people,), dtype=self.np_type)
            if int(frame_num) > 715 and int(frame_num) < 738 and seq_name == 'downtown_walking_00':
                valid[f] = np.zeros((num_people,), dtype=self.np_type)
            if int(frame_num) > 951 and int(frame_num) < 983 and seq_name == 'downtown_walking_00':
                valid[f] = np.zeros((num_people,), dtype=self.np_type)
            if int(frame_num) > 1100 and int(frame_num) < 1127 and seq_name == 'downtown_walking_00':
                valid[f] = np.zeros((num_people,), dtype=self.np_type)
            if int(frame_num) > 1172 and int(frame_num) < 1217 and seq_name == 'downtown_walking_00':
                valid[f] = np.zeros((num_people,), dtype=self.np_type)
            if int(frame_num) > 1369 and int(frame_num) < 1386 and seq_name == 'downtown_walking_00':
                valid[f] = np.zeros((num_people,), dtype=self.np_type)
        if self.sub_sequence != '' and not seq_name.startswith(self.sub_sequence): # basketball, cheers, dance, fight, handshake, highfive, hug, kiss, leg, piggy, pose, sidehug, talk
            valid = np.zeros((self.frame_length, self.max_people), dtype=self.np_type)
        load_data['valid'] = valid
        load_data['has_3d'] = has_3d
        load_data['has_smpl'] = has_smpls
        load_data['features'] = features
        load_data['verts'] = vertss
        load_data['gt_joints'] = gt_joints
        load_data['gt_joints_smpl'] = gt_joints_smpl
        # load_data['img'] = self.normalize_img(img)
        load_data['init_pose'] = init_poses
        load_data['init_pose_6d'] = init_pose_6d
        load_data['pose_6d'] = pose_6d
        load_data['pose'] = poses
        load_data['betas'] = shapes
        load_data['gt_cam_t'] = translation # frame_length, max_people, 3
        load_data['imgname'] = imgnames
        load_data['keypoints'] = keypoints
        load_data['pred_keypoints'] = pred_keypoints

        load_data["center"] = centers
        load_data["scale"] = scales
        load_data["img_h"] = img_hs
        load_data["img_w"] = img_ws
        load_data["focal_length"] = focal_lengthes
        load_data["single_person"] = single_person


        # text
        # get file path
        if self.frame_length == 16:
            seq_name = imgnames[0].split('/')[-2]
            frame_start = imgnames[0].split('/')[-1].split('.')[-2].split('_')[-1]
            frame_end = imgnames[-1].split('/')[-1].split('.')[-2].split('_')[-1]
            text_file_path = '/root/autodl-fs/PW3D/text/' + seq_name + '_' + frame_start + '_' + frame_end + '.txt'
            with open(text_file_path, 'r') as f:
                text = f.read()
                # get the 2nd line
                text1 = text.split('\n')[1]
                text1 = text1[8:]
                text1 = text1.replace('their', 'his')
                # get the 3rd line
                text2 = text.split('\n')[2]
                text2 = text2[8:]
                text2 = text2.replace('their', 'his')
            load_data['text1'] = text1
            load_data['text2'] = text2
        
        # text_comp
        # get file path
        if self.frame_length == 64 and self.is_train==False:
            seq_name = imgnames[0].split('/')[-2]
            frame_start = imgnames[0].split('/')[-1].split('.')[-2].split('_')[-1]
            frame_end = imgnames[-1].split('/')[-1].split('.')[-2].split('_')[-1]
            text_comp = '/root/autodl-tmp/PW3D/comp_seq/' + seq_name + '.pkl'
            assert os.path.exists(text_comp), f"File {text_comp} does not exist"
            with open(text_comp, 'rb') as f:
                comp_data = pickle.load(f)
                for i in range(len(comp_data['img'])):
                    # print(comp_data['img'])
                    frame_num = comp_data['img'][i].split('/')[-1].split('.')[-2].split('_')[-1]
                    if int(frame_num) == int(frame_start):
                        # print(seq_name, cam_name, frame_start, frame_end)
                        load_data['comp_pose_6d'] = torch.from_numpy(np.array(comp_data['pred_pose_6d'][i:i + self.frame_length]))
                        # print(load_data['comp_pose_6d'].shape)
                        # print(load_data['init_pose_6d'].shape)
                        if len(load_data['comp_pose_6d']) < self.frame_length:
                            load_data['comp_pose_6d'] = torch.cat((load_data['comp_pose_6d'], load_data['init_pose_6d'][len(load_data['comp_pose_6d']):]), dim=0)
                        # print(len(load_data['comp_pose_6d']))
                        assert len(load_data['comp_pose_6d']) == self.frame_length, f"Length of comp_pose_6d is {len(load_data['comp_pose_6d'])}, but should be {self.frame_length}"
                        break
        
        return load_data

    def __getitem__(self, index):
        data = self.create_data(index)
        return data

    def __len__(self):
        return self.len
    
if __name__ == '__main__':
    from utils.smpl_torch_batch import SMPLModel
    model_smpl = SMPLModel(
                            device=torch.device('cpu'),
                            model_path='./data/smpl/SMPL_NEUTRAL.pkl', 
                            data_type=torch.float32,
                        )
    test = Reconstruction_Feature_Data(train=True, dtype=torch.float32, data_folder='', name='', smpl=model_smpl, frame_length=16)
    test.create_data(index=0)
    test.__getitem__(index=0)
    print(test.__len__())
    pass













