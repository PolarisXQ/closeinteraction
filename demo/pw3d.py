import pickle
import numpy as np
import os
import torch
from utils.smpl_torch_batch import SMPLModel
# from utils.geometry import perspective_projection
from utils.rotation_conversions import matrix_to_axis_angle, axis_angle_to_matrix
from utils.renderer_pyrd import Renderer
import cv2
from utils.geometry import perspective_projection

# from CloseInt.CloseInt_core import CloseInt_Predictor
from hmr2.hmr2_core import Human4D_Predictor
from mobile_sam import SamPredictor, sam_model_registry
from AutoTrackAnything.AutoTrackAnything_core import AutoTrackAnythingPredictor, YOLO_clss
from alphapose_core.alphapose_core import AlphaPose_Predictor

sam = sam_model_registry["vit_t"](checkpoint="pretrained/AutoTrackAnything_data/mobile_sam.pt")
sam_predictor = SamPredictor(sam)

yolo_predictor = YOLO_clss('yolox')

autotrack = AutoTrackAnythingPredictor(sam_predictor, yolo_predictor)

model_dir = 'pretrained/pw3d_data/Human4D_checkpoints/epoch=35-step=1000000.ckpt'
human4d_predictor = Human4D_Predictor(model_dir)

# pretrain_model = 'pretrained/closeint_data/best_reconstruction_epoch036_60.930809.pkl'
# predictor = CloseInt_Predictor(pretrain_model)

alpha_config = R'alphapose_core/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml'
alpha_checkpoint = R'pretrained/alphapose_data/halpe26_fast_res50_256x192.pth'
alpha_thres = 0.1
alpha_predictor = AlphaPose_Predictor(alpha_config, alpha_checkpoint, alpha_thres)


smpl = SMPLModel(
            device=torch.device('cpu'),
            model_path='/home/polaris/mocap/CloseInt/data/smpl/SMPL_NEUTRAL.pkl', 
            data_type=torch.float32,
        )

dataset_name = '3DPW'
data_root = '/home/polaris/mocap/Dataset/3DPW'

pw3d_data = {'pose2ds_pred':[], 'pose2ds':[], 'imnames':[], 'img_size':[], 'intris':[], 'features':[], 'center':[], 'patch_scale':[], 'init_poses':[], 'poses':[], 'shapes':[], 'gt_cam_t':[]}

for sequence in os.listdir(os.path.join(data_root, 'imageFiles')):
    img_folder = os.path.join(data_root, 'imageFiles', sequence)
    frames = sorted(os.listdir(img_folder))
    data_path = img_folder.replace('imageFiles', 'sequenceFiles/train') + '.pkl'
    if not os.path.isfile(data_path):
        print(f'{data_path} does not exist')
        continue
    print("processing ", sequence)
    data = pickle.load(open(data_path, 'rb'), encoding='latin1')
    # Load camera parameters
    camera_param = data['cam_intrinsics']
    img = cv2.imread(os.path.join(img_folder, frames[0]))
    img_h, img_w = img.shape[:2]
    
    cam_pose = data['cam_poses'] # 30hz
    cam_pose = torch.tensor(cam_pose, dtype=torch.float32)
    # print(cam_pose.shape) # torch.Size([765, 4, 4])
    # expand to 60hz by interpolation
    # Expand cam_pose to 60Hz by duplicating each frame
    cam_pose_60Hz = cam_pose.repeat_interleave(2, dim=0)
    # print(cam_pose_60Hz.shape)  # torch.Size([1530, 4, 4])
    cam_rot = cam_pose_60Hz[:, :3, :3]
    cam_transl = cam_pose_60Hz[:,:3, 3]
    first_cam_rot = cam_rot[0]
    first_cam_transl = cam_transl[0].unsqueeze(0)


    pose_aa = np.array(data['poses_60Hz'])
    pose_aa = torch.tensor(pose_aa, dtype=torch.float32) # [num_people, frame_length, 72]
    num_agent, frame_num = pose_aa.shape[:2]
    orient_aa_in_global = pose_aa[:,:,0:3]
    orient_mtx_in_global = axis_angle_to_matrix(orient_aa_in_global.reshape(-1,3)).reshape(pose_aa.shape[0], pose_aa.shape[1], 1,3 ,3 )
    orient_mtx_in_cam = torch.matmul(orient_mtx_in_global.transpose(4,3), first_cam_rot.transpose(1,0)).transpose(4,3).reshape(-1,3,3)
    orient_aa_in_cam = matrix_to_axis_angle(orient_mtx_in_cam).reshape(pose_aa.shape[0], pose_aa.shape[1], 3)
    pose_aa[:,:,0:3] = orient_aa_in_cam
    # -> frame_lenth, num_people, 72
    pose_aa = pose_aa.permute(1,0,2)
    # print(pose_aa.shape) # torch.Size([1530, 2, 72])
    beta = np.array(data['betas'])
    beta = torch.tensor(beta, dtype=torch.float32)
    # -> frame_length, num_people, 10 by permute and expand
    beta = beta.unsqueeze(0).expand(pose_aa.shape[0], -1, -1)
    # print(beta.shape) # torch.Size([1530, 2, 10])
    transl = np.array(data['trans_60Hz'])
    transl = torch.tensor(transl, dtype=torch.float32)
    transl = transl.permute(1,0,2)
    # print(transl.shape) # torch.Size([1530, 2, 3])

    # print(first_cam_rot.shape)
    # print(first_cam_transl.shape)
    # expand to frame_length, agent_num, 3
    cam_rot = cam_rot.unsqueeze(1).expand(-1, pose_aa.shape[1], -1, -1)
    cam_transl = cam_transl.unsqueeze(1).expand(-1, pose_aa.shape[1], -1)
    # print(cam_rot.shape) # torch.Size([1530, 2, 3, 3])
    # print(cam_transl.shape) # torch.Size([1530, 2, 3])
    # convert transl to cam coordinate


    temp_transl = torch.zeros((pose_aa.shape[0]*pose_aa.shape[1], 3), dtype=torch.float32)
    temp_pose_aa = pose_aa.reshape(-1,72)

    verts, joint = smpl(beta.reshape(-1,10),temp_pose_aa,temp_transl) # [num_agent*frame_num, 6890, 3], [num_agent*frame_num, 24, 3]
    pelvis = joint[:,0,:]

    transl = transl.reshape(-1,3)
    cam_transl = cam_transl.reshape(-1,3) # frame_length*num_people, 3
    cam_rot_t = cam_rot.reshape(-1,3,3).transpose(1,2) # frame_length*num_people, 3, 3
    transl = torch.matmul(transl, first_cam_rot.transpose(1,0)) + first_cam_transl - pelvis

    verts, joint = smpl(beta.reshape(-1,10),pose_aa.reshape(-1,72),transl.reshape(-1,3),halpe= True) # [num_agent*frame_num, 6890, 3], [num_agent*frame_num, 24, 3]

    # renderer = Renderer(focal_length=camera_param[0][0], center=(camera_param[0][2], camera_param[1][2]),
    #                                 img_w=img.shape[1], img_h=img.shape[0],
    #                                 faces=smpl.faces,
    #                                 same_mesh_color=True)
    # pred_smpl = renderer.render_front_view(verts[1].unsqueeze(0).numpy(), bg_img_rgb=img)
    # cv2.imwrite('pred_smpl.png', pred_smpl)

    camera_center = torch.tensor([camera_param[0][2], camera_param[1][2]], dtype=torch.float32).unsqueeze(0).expand(num_agent*frame_num, -1) # [num_agent*frame_num, 2]
    pose2ds = perspective_projection(joint,
                            rotation=torch.eye(3, device=joint.device).unsqueeze(0).expand(num_agent*frame_num, -1, -1),
                            translation=torch.zeros(3, device=joint.device).unsqueeze(0).expand(num_agent*frame_num, -1),
                            focal_length=torch.tensor(camera_param[0][0], dtype=torch.float32).unsqueeze(0).expand(num_agent*frame_num),
                            camera_center=camera_center)

    pose2ds = torch.cat([pose2ds, torch.ones_like(pose2ds[..., :1])], dim=-1)
    pose2ds = pose2ds.view(frame_num, num_agent, 26, 3)
    
    pw3d_data['pose2ds'].append(pose2ds.numpy())
    pw3d_data['imnames'].append([os.path.join('train/courtyard_arguing_00', frame) for frame in frames])
    pw3d_data['intris'].append(torch.tensor(camera_param, dtype=torch.float32).unsqueeze(0).expand(frame_num, -1, -1).numpy())
    pw3d_data['poses'].append(pose_aa.numpy())
    pw3d_data['shapes'].append(beta.numpy())
    pw3d_data['gt_cam_t'].append(transl.numpy())
    
    results, total_person = autotrack.inference(img_folder, viz=False)

    # Initialize temporary lists
    features_list = []
    init_poses_list = []
    center_list = []
    patch_scale_list = []
    pose2ds_pred_list = []
    img_size_list = []

    for frame, bbox in zip(frames, results):
        img = os.path.join(img_folder, frame)
        bbox = np.array([bbox[key] for key in bbox.keys()])
        params = human4d_predictor.closeint_data(img, bbox, viz=False)

        img = cv2.imread(img)
        img_h, img_w = img.shape[:2]
        pose = alpha_predictor.predict(img, bbox)

        # Append to temporary lists instead of pw3d_data
        features_list.append(params['features'])
        init_poses_list.append(params['pose'])
        center_list.append(params['centers'])
        patch_scale_list.append(params['scales'])
        pose2ds_pred_list.append(pose)
        img_size_list.append([img_h, img_w] * len(bbox))
        # pw3d_data['focal_length'].append([focal_length]*len(bbox))
        
    # Append all lists to pw3d_data
    pw3d_data['features'].append(features_list)
    pw3d_data['init_poses'].append(init_poses_list)
    pw3d_data['center'].append(center_list)
    pw3d_data['patch_scale'].append(patch_scale_list)
    pw3d_data['pose2ds_pred'].append(pose2ds_pred_list)
    pw3d_data['img_size'].append(img_size_list)


# save pw3d_data to a pkl file
for key in pw3d_data.keys():
    print(key, len(pw3d_data[key]))
    
import pickle
with open('pw3d_data.pkl', 'wb') as f:
    pickle.dump(pw3d_data, f)