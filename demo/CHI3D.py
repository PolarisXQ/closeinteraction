import sys
sys.path.append('./')
import os
import cv2
import numpy as np
import json
from utils.smpl_torch_batch import SMPLModel
from utils.geometry import perspective_projection
from utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix, matrix_to_rotation_6d
from utils.renderer_pyrd import Renderer

import torch

smpl = SMPLModel(
            device=torch.device('cpu'),
            model_path='/root/CloseInt/data/smpl/SMPL_NEUTRAL.pkl', 
            data_type=torch.float32,
        )

img_folder = '/root/autodl-tmp/CHI3D/images/train/s02/50591643/Grab 1'

frames = sorted(os.listdir(img_folder))

human4d_data = {'pose2ds_pred':[], 'pose2ds':[], 'imnames':[], 'img_size':[], 'intris':[], 'features':[], 'center':[], 'patch_scale':[], 'init_poses':[], 'poses':[], 'shapes':[], 'gt_cam_t':[]}
human4d_data['imnames'].append([os.path.join('s02/50591643/Grab1', frame) for frame in frames])
# load from json file
camera_param = json.load(open('/root/autodl-fs/CHI3D/train/s02/camera_parameters/50591643/Grab 1.json'))
# load the first image to get the image size
img = cv2.imread(os.path.join(img_folder, frames[0]))
img_h, img_w = img.shape[:2]

smplx_param_data = json.load(open('/root/autodl-fs/CHI3D/train/s02/smplx/Grab 1.json'))

pose_mtx = smplx_param_data['body_pose']
pose_mtx = torch.tensor(pose_mtx, dtype=torch.float32)
num_agent, frame_num, joint_num = pose_mtx.shape[:3]
pose_mtx = pose_mtx.permute(1,0,2,3,4) # [frame_num, num_agent, 21, 3, 3]
# create identity rotation mtrix for hands pose
eye = torch.eye(3, dtype=torch.float32) # [3, 3]
# repeat the identity matrix for each agent and frame and 2 hands
eye = eye.unsqueeze(0).unsqueeze(0).expand(frame_num, num_agent, 2, 3, 3) # [num_agent, frame_num, 2, 3, 3]
orient = smplx_param_data['global_orient']
orient = torch.tensor(orient, dtype=torch.float32) # [num_agent, frame_num, 1, 3, 3]
orient = orient.permute(1,0,2,3,4) # [frame_num, num_agent, 1, 3, 3]
extrinsics_Rt = torch.tensor(camera_param['extrinsics']['R'], dtype=torch.float32).transpose(1,0) # [num_agent*frame_num, 3, 3]
cam_orient = torch.matmul(orient.transpose(4,3),extrinsics_Rt).transpose(4,3) # [num_agent, frame_num, 1, 3, 3]
pose_mtx = torch.cat([cam_orient, pose_mtx, eye], dim=2) # [num_agent, frame_num, 24, 3]
pose_aa = matrix_to_axis_angle(pose_mtx.view(-1, 3, 3)).view(frame_num, num_agent, 24, 3) # [frame_num, num_agent, 24, 3]
human4d_data['poses'].append(pose_aa.numpy())
betas = smplx_param_data['betas']
betas = torch.tensor(betas, dtype=torch.float32) # [num_agent, frame_num, 10]
betas = betas.permute(1,0,2) # [frame_num, num_agent, 10]
human4d_data['shapes'].append(betas.numpy())
transl = smplx_param_data['transl']
transl = torch.tensor(transl, dtype=torch.float32) # [num_agent, frame_num, 3]
transl = transl.permute(1,0,2).reshape(-1, 3) # [frame_num*num_agent, 3]
temp_trans = torch.zeros((frame_num*num_agent, 3), dtype=torch.float32)
temp_pose_aa = torch.zeros((frame_num*num_agent, 24, 3), dtype=torch.float32)
verts, joint = smpl(betas.reshape(-1,10), temp_pose_aa.reshape(-1,72),temp_trans) # [num_agent*frame_num, 6890, 3], [num_agent*frame_num, 24, 3]
pelvis = joint[:,0,:] # [num_agent*frame_num, 3]
extrinsics_T = torch.tensor(camera_param['extrinsics']['T'], dtype=torch.float32) # [num_agent*frame_num, 3]
print(transl.shape, pelvis.shape, extrinsics_T.shape, extrinsics_Rt.shape)
cam_t = torch.matmul(transl + pelvis - extrinsics_T, extrinsics_Rt) - pelvis # [num_agent*frame_num, 3]
print("cam_t[0]", cam_t[0])
# cam_t[0] = torch.tensor([5.9300e-01, -1.1052e-03 , 3.1986e+00], dtype=torch.float32)
verts, joint = smpl(betas.reshape(-1,10), pose_aa.reshape(-1,72),cam_t.reshape(-1,3), halpe=True) # [num_agent*frame_num, 6890, 3], [num_agent*frame_num, 24, 3]
human4d_data['gt_cam_t'].append(cam_t.view(frame_num, num_agent, 3).numpy())
intrinsics = camera_param['intrinsics_wo_distortion']['f'] + camera_param['intrinsics_wo_distortion']['c']
intris = torch.tensor([[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]], [0, 0, 1]], dtype=torch.float32)
# repeat the intrinsics for each frame and num agent, frame
intris = intris.unsqueeze(0).expand(frame_num, num_agent, -1, -1) # [frame_num, num_agent, 3, 3]
# convert to array
human4d_data['intris'].append(intris.numpy())

camera_center = torch.stack([torch.tensor([intrinsics[2], intrinsics[3]], dtype=torch.float32)]*frame_num*num_agent) # [num_agent*frame_num, 2]
pose2ds = perspective_projection(joint + cam_t[:, None, :],
                                rotation=torch.eye(3, device=joint.device).unsqueeze(0).expand(num_agent*frame_num, -1, -1),
                                translation=torch.zeros(3, device=joint.device).unsqueeze(0).expand(num_agent*frame_num, -1),
                                focal_length=intris.reshape(-1, 3,3)[:,0,0],
                                camera_center=camera_center)
# append ones to last dimension
pose2ds = torch.cat([pose2ds, torch.ones_like(pose2ds[..., :1])], dim=-1)
pose2ds = pose2ds.view(frame_num, num_agent, 26, 3)
human4d_data['pose2ds'].append(pose2ds.numpy())

renderer = Renderer(focal_length=camera_param['intrinsics_wo_distortion']['f'][0], center=camera_center[0],
                    img_w=img.shape[1], img_h=img.shape[0],
                    faces=smpl.faces,
                    same_mesh_color=True)
pred_smpl = renderer.render_front_view(verts[0].unsqueeze(0).numpy(), bg_img_rgb=img)
pred_smpl = cv2.cvtColor(pred_smpl, cv2.COLOR_BGR2RGB)
cv2.imwrite('pred_smpl.png', pred_smpl)
for key in human4d_data.keys():
    print(key, len(human4d_data[key]))

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

# results, total_person = autotrack.inference(img_folder, viz=False)

# # Initialize temporary lists
# features_list = []
# init_poses_list = []
# center_list = []
# patch_scale_list = []
# pose2ds_pred_list = []
# img_size_list = []

# for frame, bbox in zip(frames, results):
#     img = os.path.join(img_folder, frame)
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

# # save human4d_data to a pkl file
# for key in human4d_data.keys():
#     print(key, len(human4d_data[key]))
# import pickle
# with open('human4d_data.pkl', 'wb') as f:
#     pickle.dump(human4d_data, f)