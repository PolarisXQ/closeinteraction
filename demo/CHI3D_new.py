import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from util.dataset_util import read_data, project_3d_to_2d, plot_over_image
from util.signature_util import SignatureVisualizer
import os
import torch
from utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix, matrix_to_rotation_6d
from utils.renderer_pyrd import Renderer
from utils.smpl_torch_batch import SMPLModel
from utils.geometry import perspective_projection

# # from CloseInt.CloseInt_core import CloseInt_Predictor
# from hmr2.hmr2_core import Human4D_Predictor
# from mobile_sam import SamPredictor, sam_model_registry
# from AutoTrackAnything.AutoTrackAnything_core import AutoTrackAnythingPredictor, YOLO_clss
# from alphapose_core.alphapose_core import AlphaPose_Predictor

# sam = sam_model_registry["vit_t"](checkpoint="pretrained/AutoTrackAnything_data/mobile_sam.pt")
# sam_predictor = SamPredictor(sam)

# yolo_predictor = YOLO_clss('yolox')

# autotrack = AutoTrackAnythingPredictor(sam_predictor, yolo_predictor)

# model_dir = 'pretrained/Human4D_data/Human4D_checkpoints/epoch=35-step=1000000.ckpt'
# human4d_predictor = Human4D_Predictor(model_dir)

# # pretrain_model = 'pretrained/closeint_data/best_reconstruction_epoch036_60.930809.pkl'
# # predictor = CloseInt_Predictor(pretrain_model)

# alpha_config = R'alphapose_core/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml'
# alpha_checkpoint = R'pretrained/alphapose_data/halpe26_fast_res50_256x192.pth'
# alpha_thres = 0.1
# alpha_predictor = AlphaPose_Predictor(alpha_config, alpha_checkpoint, alpha_thres)
            
smpl = SMPLModel(
            device=torch.device('cpu'),
            model_path='/root/CloseInt/data/smpl/SMPL_NEUTRAL.pkl', 
            data_type=torch.float32,
        )

# path to the parent directory (contains test/ meta/)
dataset_name = 'CHI3D'
data_root = '/root/autodl-fs'
subset = 'train'
SMPLX_Models_Path = '/root/CloseInt/data'
img_root_folder = '/root/autodl-tmp/CHI3D/images/train'

from util.smplx_util import SMPLXHelper
smplx_helper = SMPLXHelper(SMPLX_Models_Path)

human4d_data = {'pose2ds_pred':[], 'pose2ds':[], 'imnames':[], 'img_size':[], 'intris':[], 'features':[], 'center':[], 'patch_scale':[], 'init_poses':[], 'poses':[], 'shapes':[], 'gt_cam_t':[]}


for subj_name in os.listdir(img_root_folder):
    if subj_name != 's02':
        continue
    subj_folder = os.path.join(img_root_folder, subj_name)
    if not os.path.isdir(subj_folder):
        continue

    for camera_name in os.listdir(subj_folder):
        camera_folder = os.path.join(subj_folder, camera_name)
        if not os.path.isdir(camera_folder):
            continue
        if subj_name == 's04' and (camera_name == '60457274' or camera_name == '65906101'): # remain for test
            continue

        for action_name in os.listdir(camera_folder):

            action_folder = os.path.join(camera_folder, action_name)
            if not os.path.isdir(action_folder):
                continue
            print("processing ", subj_name, camera_name, action_name)
            frames, j3ds1, cam_params, gpp_data1, smplx_param_data1, annotations1 = read_data(data_root, 
                                                                                        dataset_name, 
                                                                                        subset, 
                                                                                        subj_name, 
                                                                                        action_name, 
                                                                                        camera_name,
                                                                                        subject='w_markers')

            frames, j3ds2, cam_params, gpp_data2, smplx_param_data2, annotations2 = read_data(data_root, 
                                                                                        dataset_name, 
                                                                                        subset, 
                                                                                        subj_name, 
                                                                                        action_name, 
                                                                                        camera_name,
                                                                                        subject='wo_markers')

            world_smplx_params1 = smplx_helper.get_world_smplx_params(smplx_param_data1)
            camera_smplx_params1 = smplx_helper.get_camera_smplx_params(smplx_param_data1, cam_params)
            world_smplx_params2 = smplx_helper.get_world_smplx_params(smplx_param_data2)
            camera_smplx_params2 = smplx_helper.get_camera_smplx_params(smplx_param_data2, cam_params)

            # frames = sorted(os.listdir(action_folder))
            # human4d_data['imnames'].append([os.path.join(f'images/train/{subj_name}/{camera_name}/{action_name}', frame) for frame in frames])
            camera_param = json.load(open(f'/root/autodl-fs/CHI3D/train/{subj_name}/camera_parameters/{camera_name}/{action_name}.json'))
            intrinsics = camera_param['intrinsics_wo_distortion']['f'] + camera_param['intrinsics_wo_distortion']['c']
            intris = torch.tensor([[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]], [0, 0, 1]], dtype=torch.float32)
            frame_num = world_smplx_params1['betas'].shape[0]
            num_agent = 2
            # intriss = intris.unsqueeze(0).expand(frame_num, num_agent, -1, -1) # [frame_num, num_agent, 3, 3]
            # human4d_data['intris'].append(intriss.numpy())

            poses = torch.cat([camera_smplx_params1['body_pose'].unsqueeze(1), camera_smplx_params2['body_pose'].unsqueeze(1)], dim=1) # [frame_num, num_agent, 21, 3, 3]
            eye = torch.eye(3, dtype=torch.float32) # [3, 3]
            eye = eye.unsqueeze(0).unsqueeze(0).expand(frame_num, num_agent, 2, 3, 3) # [num_agent, frame_num, 2, 3, 3]
            orient = torch.cat([camera_smplx_params1['global_orient'], camera_smplx_params2['global_orient']], dim=1).unsqueeze(2) # [num_agent, frame_num, 2, 3, 3]
            pose_mtx = torch.cat([orient, poses, eye], dim = 2)
            pose_aa = matrix_to_axis_angle(pose_mtx.view(-1, 3, 3)).view(frame_num, num_agent, 24, 3) # [frame_num, num_agent, 24, 3]
            # human4d_data['poses'].append(pose_aa.numpy())

            betas = torch.cat([camera_smplx_params1['betas'].unsqueeze(1), camera_smplx_params2['betas'].unsqueeze(1)], dim=1) # [frame_num, num_agent, 10]
            # print(betas.shape)
            # human4d_data['shapes'].append(betas.numpy())

            transl = torch.cat([camera_smplx_params1['transl'], camera_smplx_params2['transl']], dim=1) # [frame_num, num_agent, 3]
            human4d_data['gt_cam_t'].append(transl.view(frame_num, num_agent, 3).numpy())

            # verts, joint = smpl(betas.reshape(-1,10), pose_aa.reshape(-1,72), transl.reshape(-1,3), halpe=True) # [num_agent*frame_num, 6890, 3], [num_agent*frame_num, 24, 3]
            joints1 = smplx_helper.smplx_model(**camera_smplx_params1).joints[:,:21,:] # [frame_num, 24, 3]
            joints2 = smplx_helper.smplx_model(**camera_smplx_params2).joints[:,:21,:] # [frame_num, 24, 3]
            print(joints1.shape, joints2.shape)
            joint = torch.cat([joints1.unsqueeze(1), joints2.unsqueeze(1)], dim=1).reshape(-1, 21, 3)
            print(joint.shape)

            camera_center = torch.tensor([intris[0,2], intris[1,2]], dtype=torch.float32).unsqueeze(0).expand(num_agent*frame_num, -1) # [num_agent*frame_num, 2]
            pose2ds = perspective_projection(joint,
                                            rotation=torch.eye(3, device=joint.device).unsqueeze(0).expand(num_agent*frame_num, -1, -1),
                                            translation=torch.zeros(3, device=joint.device).unsqueeze(0).expand(num_agent*frame_num, -1),
                                            focal_length=intris.reshape(-1, 3,3)[:,0,0],
                                            camera_center=camera_center)

            # append ones to last dimension
            pose2ds = torch.cat([pose2ds, torch.ones_like(pose2ds[..., :1])], dim=-1)
            pose2ds = pose2ds.view(frame_num, num_agent, 21, 3)
            human4d_data['pose2ds'].append(pose2ds.numpy())

            # results, total_person = autotrack.inference(action_folder, viz=False)

            # # Initialize temporary lists
            # features_list = []
            # init_poses_list = []
            # center_list = []
            # patch_scale_list = []
            # pose2ds_pred_list = []
            # img_size_list = []

            # for frame, bbox in zip(frames, results):
            #     img = os.path.join(action_folder, frame)
            #     bbox = np.array([bbox[key] for key in bbox.keys()])
            #     params = human4d_predictor.closeint_data(img, bbox, viz=False)

            #     img = cv2.imread(img)
            #     img_h, img_w = img.shape[:2]
            #     pose = alpha_predictor.predict(img, bbox)

            #     # Append to temporary lists instead of human4d_data
            #     features_list.append(params['features'])
            #     init_poses_list.append(params['pose'])
            #     center_list.append(params['centers'])
            #     patch_scale_list.append(params['scales'])
            #     pose2ds_pred_list.append(pose)
            #     img_size_list.append([img_h, img_w] * len(bbox))
            #     # human4d_data['focal_length'].append([focal_length]*len(bbox))
                
            # # Append all lists to human4d_data
            # human4d_data['features'].append(features_list)
            # human4d_data['init_poses'].append(init_poses_list)
            # human4d_data['center'].append(center_list)
            # human4d_data['patch_scale'].append(patch_scale_list)
            # human4d_data['pose2ds_pred'].append(pose2ds_pred_list)
            # human4d_data['img_size'].append(img_size_list)

# save human4d_data to a pkl file
for key in human4d_data.keys():
    print(key, len(human4d_data[key]))
    
import pickle
# with open('human4d_data.pkl', 'wb') as f:
#     pickle.dump(human4d_data, f)

with open('/root/autodl-tmp/CHI3D/annots/chi3d_s02.pkl', 'rb') as file:
    old_data = pickle.load(file)
    
old_data['gt_cam_t'] = human4d_data['gt_cam_t']
old_data['pose2ds'] = human4d_data['pose2ds']

# save updated old_data to a pkl file
with open('human4d_data.pkl', 'wb') as f:
    pickle.dump(old_data, f)
