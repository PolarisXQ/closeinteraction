import pickle
import numpy as np
import os
import sys
# from utils.smpl_torch_batch import SMPLModel
import pickle
# from utils.rotation_conversions import matrix_to_axis_angle, axis_angle_to_matrix
# from utils.renderer_pyrd import Renderer
import cv2
# from utils.geometry import perspective_projection

# from CloseInt.CloseInt_core import CloseInt_Predictor
from hmr2.hmr2_core import Human4D_Predictor
from mobile_sam import SamPredictor, sam_model_registry
from AutoTrackAnything.AutoTrackAnything_core import AutoTrackAnythingPredictor, YOLO_clss
from alphapose_core.alphapose_core import AlphaPose_Predictor

sam = sam_model_registry["vit_t"](checkpoint="pretrained/AutoTrackAnything_data/mobile_sam.pt")
sam_predictor = SamPredictor(sam)

yolo_predictor = YOLO_clss('yolox')

autotrack = AutoTrackAnythingPredictor(sam_predictor, yolo_predictor)

model_dir = 'pretrained/Human4D_data/Human4D_checkpoints/epoch=35-step=1000000.ckpt'
human4d_predictor = Human4D_Predictor(model_dir)

# pretrain_model = 'pretrained/closeint_data/best_reconstruction_epoch036_60.930809.pkl'
# predictor = CloseInt_Predictor(pretrain_model)

alpha_config = R'alphapose_core/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml'
alpha_checkpoint = R'pretrained/alphapose_data/halpe26_fast_res50_256x192.pth'
alpha_thres = 0.1
alpha_predictor = AlphaPose_Predictor(alpha_config, alpha_checkpoint, alpha_thres)


# smpl = SMPLModel(
#             device=torch.device('cpu'),
#             model_path='/home/polaris/mocap/CloseInt/data/smpl/SMPL_NEUTRAL.pkl', 
#             data_type=torch.float32,
#         )

data_root = '/root/autodl-tmp/test'
output_path = '/root/autodl-tmp/4dhuman_data_test'

def process_data(sequence):
    img_folder = os.path.join(data_root, sequence, 'exo')
    for cam in os.listdir(img_folder):
        try:
        # if True:
            alpha_predictor = AlphaPose_Predictor(alpha_config, alpha_checkpoint, alpha_thres)
            # check if the pkl file exists
            if os.path.isfile(os.path.join(output_path, f'{sequence}_{cam}.pkl')):
                print(f'{sequence}_{cam}.pkl already exists')
                continue
            img_folder_cam = os.path.join(img_folder, cam, 'images')
            frames = sorted(os.listdir(img_folder_cam))
            # remove things not end with .jpg
            frames = [frame for frame in frames if frame.endswith('.jpg')]
            print("processing ", img_folder_cam)
            
            results, total_person = autotrack.inference(img_folder_cam, viz=False)

            # Initialize temporary lists
            features_list = []
            init_poses_list = []
            center_list = []
            patch_scale_list = []
            pose2ds_pred_list = []
            img_size_list = []

            for frame, bbox in zip(frames, results):
                img = os.path.join(img_folder_cam, frame)
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
                
            # save the data to pw3d_data
            # pw3d_data['features'].append(features_list)
            # pw3d_data['init_poses'].append(init_poses_list)
            # pw3d_data['center'].append(center_list)
            # pw3d_data['patch_scale'].append(patch_scale_list)
            # pw3d_data['pose2ds_pred'].append(pose2ds_pred_list)
            # pw3d_data['img_size'].append(img_size_list)
            pw3d_data = {
                'features': features_list,
                'init_poses': init_poses_list,
                'center': center_list,
                'patch_scale': patch_scale_list,
                'pose2ds_pred': pose2ds_pred_list,
                'img_size': img_size_list
            }


            # save pw3d_data to a pkl file
            for key in pw3d_data.keys():
                print(key, len(pw3d_data[key]))
            
            with open(os.path.join(output_path, f'{sequence}_{cam}.pkl'), 'wb') as f:
                pickle.dump(pw3d_data, f)
            print(f'saved {sequence}_{cam}.pkl')
        except Exception as e:
            print(f"Error processing {sequence} {cam}: {e}")
            # write the error to a file
            with open('error.txt', 'a') as f:
                f.write(f"Error processing {sequence}_{cam}: {e}\n")
            continue
        
if __name__ == '__main__':
    # sequence = sys.argv[1]
    for sequence in os.listdir(data_root):
        if '0' in sequence:
            if os.path.isdir(os.path.join(data_root, sequence)):
                process_data(sequence)
    # sequence = '004_karate'
    # process_data(sequence)