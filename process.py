'''
 @FileName    : process.py
 @EditTime    : 2022-09-27 16:18:51
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def merge_gt(data):
    batch_size, frame_length, agent_num = data['pose'].shape[:3]

    data['data_shape'] = data['pose'].shape[:3]
    data['has_3d'] = data['has_3d'].reshape(batch_size*frame_length*agent_num,1)
    data['has_smpl'] = data['has_smpl'].reshape(batch_size*frame_length*agent_num,1)
    data['verts'] = data['verts'].reshape(batch_size*frame_length*agent_num, 6890, 3)
    data['gt_joints'] = data['gt_joints'].reshape(batch_size*frame_length*agent_num, -1, 4)
    data['gt_joints_smpl'] = data['gt_joints_smpl'].reshape(batch_size*frame_length*agent_num, -1, 3)
    data['pose'] = data['pose'].reshape(batch_size*frame_length*agent_num, 72)
    data['betas'] = data['betas'].reshape(batch_size*frame_length*agent_num, 10)
    data['gt_cam_t'] = data['gt_cam_t'].reshape(batch_size*frame_length*agent_num, 3)
    data['x'] = data['x'].reshape(batch_size*frame_length*agent_num, -1)

    imgname = (np.array(data['imgname']).T).reshape(batch_size*frame_length,)
    data['imgname'] = imgname.tolist()

    return data

def extract_valid(data):
    batch_size, frame_length, agent_num = data['keypoints'].shape[:3]

    data['data_shape'] = data['keypoints'].shape[:3]
    data['center'] = data['center'].reshape(batch_size*frame_length*agent_num, -1)
    data['scale'] = data['scale'].reshape(batch_size*frame_length*agent_num,)
    data['img_h'] = data['img_h'].reshape(batch_size*frame_length*agent_num,)
    data['img_w'] = data['img_w'].reshape(batch_size*frame_length*agent_num,)
    data['focal_length'] = data['focal_length'].reshape(batch_size*frame_length*agent_num,)
    data['valid'] = data['valid'].reshape(batch_size*frame_length*agent_num,)

    data['has_3d'] = data['has_3d'].reshape(batch_size*frame_length*agent_num,1)
    data['has_smpl'] = data['has_smpl'].reshape(batch_size*frame_length*agent_num,1)
    data['verts'] = data['verts'].reshape(batch_size*frame_length*agent_num, -1, 3)
    # data['contact_correspondences'] = data['contact_correspondences
    data['gt_joints'] = data['gt_joints'].reshape(batch_size*frame_length*agent_num, -1, data['gt_joints'].shape[-1])
    data['pose'] = data['pose'].reshape(batch_size*frame_length*agent_num, 72)
    data['betas'] = data['betas'].reshape(batch_size*frame_length*agent_num, 10)
    data['keypoints'] = data['keypoints'].reshape(batch_size*frame_length*agent_num, -1, data['keypoints'].shape[-1])
    data['pred_keypoints'] = data['pred_keypoints'].reshape(batch_size*frame_length*agent_num, -1, data['pred_keypoints'].shape[-1])
    data['gt_cam_t'] = data['gt_cam_t'].reshape(batch_size*frame_length*agent_num, 3)

    imgname = (np.array(data['imgname']).T).reshape(batch_size*frame_length,)
    data['imgname'] = imgname.tolist()
    if 'text1' in data.keys():
        text1 = (np.array(data['text1']).T).reshape(batch_size,)
        data['text1'] = text1.tolist()
        text2 = (np.array(data['text2']).T).reshape(batch_size,)
        data['text2'] = text2.tolist()

    return data

def extract_valid_demo(data):
    batch_size, agent_num, _, _, _ = data['norm_img'].shape
    valid = data['valid'].reshape(-1,)

    data['center'] = data['center'].reshape(batch_size*agent_num, -1)[valid == 1]
    data['scale'] = data['scale'].reshape(batch_size*agent_num,)[valid == 1]
    data['img_h'] = data['img_h'].reshape(batch_size*agent_num,)[valid == 1]
    data['img_w'] = data['img_w'].reshape(batch_size*agent_num,)[valid == 1]
    data['focal_length'] = data['focal_length'].reshape(batch_size*agent_num,)[valid == 1]

    # imgname = (np.array(data['imgname']).T).reshape(batch_size*agent_num,)[valid.detach().cpu().numpy() == 1]
    # data['imgname'] = imgname.tolist()

    return data

def to_device(data, device):
    imnames = {'imgname':data['imgname']} 
    if 'text1' in data.keys():
        text1 = {'text1':data['text1']}
        text2 = {'text2':data['text2']}
        data = {k:v.to(device).float() for k, v in data.items() if k not in ['imgname','text1','text2']}
        data = {**imnames, **text1, **text2, **data}
    else:
        data = {k:v.to(device).float() for k, v in data.items() if k not in ['imgname']}
        data = {**imnames, **data}
    return data

def reconstruction_train(model, loss_func, train_loader, epoch, num_epoch, device=torch.device('cpu'), **kwargs):
    writer = SummaryWriter(model.output)
    print('-' * 10 + 'model training' + '-' * 10)
    len_data = len(train_loader)
    model.model.train(mode=True)
    if model.scheduler is not None:
        model.scheduler.step()

    train_loss = 0.
    for i, data in enumerate(train_loader):
        # if i > 10:
        #     break
        batchsize = data['keypoints'].shape[0]
        data = to_device(data, device)
        data = extract_valid(data)

        # forward
        pred = model.model(data)

        # calculate loss
        loss, cur_loss_dict = loss_func.calcul_trainloss(pred, data)

        debug = False
        if debug:
            results = {}
            results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(pred_pose=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
            results.update(pred_shape=pred['pred_shape'].detach().cpu().numpy().astype(np.float32))
            results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_pose=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_shape=pred['pred_shape'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
            model.save_generated_interaction(results, i, batchsize)

        debug = False
        if debug:
            results = {}
            results.update(valid=data['valid'])
            results.update(imgs=data['imgname'])
            results.update(single_person=data['single_person'])
            results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
            if 'pred_verts' not in pred.keys():
                results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_joints=data['gt_joints'].detach().cpu().numpy().astype(np.float32))
                model.save_joint_results(results, i, batchsize)
            else:
                # if 'init_verts' in pred.keys():
                #     results.update(init_verts=pred['init_verts'].detach().cpu().numpy().astype(np.float32))
                # if 'comp_verts' in pred.keys():
                #     results.update(comp_verts=pred['comp_verts'].detach().cpu().numpy().astype(np.float32))
                # results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
                # results.update(gt_verts=data['verts'].detach().cpu().numpy().astype(np.float32))
                # model.save_results(results, i, batchsize, False)
                results.update(pred_keypoints_2d = pred['pred_keypoints_2d'].detach().cpu().numpy().astype(np.float32))
                results.update(psudu_gt_keypoints_2d = data['pred_keypoints'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_keypoints_2d = data['keypoints'].detach().cpu().numpy().astype(np.float32))
                results.update(center = data['center'].detach().cpu().numpy().astype(np.float32))
                model.save_2djoints(results, i, batchsize, False)

        # backward
        model.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(parameters=model.model.parameters(), max_norm=100, norm_type=2)

        # optimize
        model.optimizer.step()
        if model.scheduler is not None:
            model.scheduler.batch_step()

        loss_batch = loss.detach()
        print('epoch: %d/%d, batch: %d/%d, loss: %.6f' %(epoch, num_epoch, i, len_data, loss_batch), cur_loss_dict)
        writer.add_scalar('Loss/train', loss_batch, epoch * len_data + i)
        for key, value in cur_loss_dict.items():
            writer.add_scalar(f'LossDict/train/{key}', value, epoch * len_data + i)
        train_loss += loss_batch
    writer.close()
    return train_loss/len_data

def extract_sliding_window(pred, data, datashape, window_size=3, last_window=False):
    """
    Keep the sliding window size of the data by setting 'valid' after window_size to zero
    """

    batchsize, frame_length, agent_num = datashape
    for i in range(batchsize):
        if not last_window[i]:
            data['valid'] = data['valid'].reshape(batchsize, frame_length, agent_num)
            data['valid'][i, window_size:] = 0 
            data['valid'] = data['valid'].reshape(batchsize*frame_length*agent_num,)
    return pred, data

def reconstruction_test(model, loss_func, loader, epoch, device=torch.device('cpu'), **kwargs):
    print('-' * 10 + 'model testing' + '-' * 10)
    loss_all = 0.
    valid_sum = 0
    loss_dict_all = {}
    model.model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            # if i != 0 and i !=12:
            #     continue
            # if i !=12 and i != 0:
            #     continue
            batchsize, frame_length, agent_num = data['keypoints'].shape[:3]
            data = to_device(data, device)
            data = extract_valid(data)

            # forward
            pred = model.model(data)
            
            # seq_len = data['seq_len']
            # idx = data['idx']
            # last_window = (idx + frame_length + 3) >= seq_len
            
            # pred, data = extract_sliding_window(pred, data, [batchsize, frame_length, agent_num], window_size=3, last_window=last_window)

            # calculate loss
            loss, cur_loss_dict = loss_func.calcul_testloss(pred, data)
            
            if False: #loss.max() > 100:
                results = {}
                results.update(imgs=data['imgname'])
                results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_pose=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_shape=pred['pred_shape'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_pose=data['pose'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_shape=data['betas'].detach().cpu().numpy().astype(np.float32))
                results.update(img_h=data['img_h'].detach().cpu().numpy().astype(np.float32))
                results.update(img_w=data['img_w'].detach().cpu().numpy().astype(np.float32))
                results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
                model.save_params(results, i, batchsize)


            if False: # loss<130: #loss.max() > 100:
                results = {}
                results.update(valid=data['valid'])
                results.update(imgs=data['imgname'])
                results.update(single_person=data['single_person'])
                results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
                if 'text1' in data.keys():
                    results.update(text1=data['text1'])
                    results.update(text2=data['text2'])
                if 'MPJPE_Instance' in cur_loss_dict.keys():
                    results.update(MPJPE=cur_loss_dict['MPJPE_Instance'].detach().cpu().numpy().astype(np.float32))
                    results.update(MPJPE_wrt_one=cur_loss_dict['MPJPE_Instance_wrt_one'].detach().cpu().numpy().astype(np.float32))
                    # MPJPE_Instance=cur_loss_dict['MPJPE_Instance']
                    # save to a npy file
                    # np.save(f'mpjpe_{i}.npy', MPJPE_Instance.detach().cpu().numpy().astype(np.float32))
                # if 'Vel_Loss_Instance' in cur_loss_dict.keys():
                #     Vel_Loss_Instance=cur_loss_dict['Vel_Loss_Instance']
                #     # save to a npy file
                #     np.save(f'vel_loss_{i}.npy', Vel_Loss_Instance.detach().cpu().numpy().astype(np.float32))
                if 'pred_verts' not in pred.keys():
                    results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_joints=data['gt_joints'].detach().cpu().numpy().astype(np.float32))
                    model.save_joint_results(results, i, batchsize)
                else:
                    if 'init_verts' in pred.keys():
                        results.update(init_verts=pred['init_verts'].detach().cpu().numpy().astype(np.float32))
                    if 'comp_verts' in pred.keys():
                        results.update(comp_verts=pred['comp_verts'].detach().cpu().numpy().astype(np.float32))
                    results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_verts=data['verts'].detach().cpu().numpy().astype(np.float32))
                    model.save_results(results, i, batchsize, True)
                    # results.update(pred_keypoints_2d = pred['pred_keypoints_2d'].detach().cpu().numpy().astype(np.float32))
                    # results.update(psudu_gt_keypoints_2d = data['pred_keypoints'].detach().cpu().numpy().astype(np.float32))
                    # results.update(gt_keypoints_2d = data['keypoints'].detach().cpu().numpy().astype(np.float32))
                    # results.update(center = data['center'].detach().cpu().numpy().astype(np.float32))
                    # model.save_2djoints(results, i, batchsize, False)

                    # results = {}
                    # results.update(valid=data['valid'])
                    # results.update(imgs=data['imgname'])
                    # results.update(single_person=data['single_person'])
                    # results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                    # results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
                    # results.update(gt_verts=data['verts'].detach().cpu().numpy().astype(np.float32))
                    # # results.update(pred_verts_contact=pred['afford_full'].detach().cpu().numpy().astype(np.float32))
                    # results.update(gt_verts_contact=data['contact_correspondences'].detach().cpu().numpy().astype(np.float32))
                    # model.save_contact_results(results, i, batchsize, True)

            loss_batch = loss.mean().detach() #/ batchsize
            # delete instance in cur_loss_dict
            if 'MPJPE_Instance' in cur_loss_dict.keys():
                cur_loss_dict.pop('MPJPE_Instance', None)
            if 'MPJPE_Instance_wrt_one' in cur_loss_dict.keys():
                cur_loss_dict.pop('MPJPE_Instance_wrt_one', None)
            print('batch: %d/%d, loss: %.6f ' %(i, len(loader), loss_batch), cur_loss_dict)
            loss_all += loss_batch
            valid_sum += data['valid'].sum().detach().detach().cpu().numpy()
            for key, value in cur_loss_dict.items():
                if key not in loss_dict_all.keys():
                    loss_dict_all[key] = [value]
                else:
                    loss_dict_all[key].append(value)
        loss_all = loss_all / valid_sum
        # cal mean of loss_dict_all
        for key, value in loss_dict_all.items():
            loss_dict_all[key] = np.sum(value) / valid_sum
        return loss_all, loss_dict_all
    
import pickle  
def save_pred(model, loss_func, loader, epoch, device=torch.device('cpu'), **kwargs):
    print('-' * 10 + 'save model output' + '-' * 10)
    model.model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            batchsize, frame_length, agent_num = data['keypoints'].shape[:3]
            data = to_device(data, device)
            data = extract_valid(data)

            # forward
            pred = model.model(data)
            
            # save to a pkl file
            results = {}
            results.update(imgs=data['imgname'])
            results.update(pred_pose_6d=pred['pred_pose6d'].detach().cpu().numpy().astype(np.float32))
            results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            # results.update(pred_joints_smpl=pred['pred_joints_smpl'].detach().cpu().numpy().astype(np.float32))
            results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
            # results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
            # results.update(gt_joints_smpl=data['gt_joints_smpl'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_joints=data['gt_joints'].detach().cpu().numpy().astype(np.float32))
            # results.update(gt_verts=data['verts'].detach().cpu().numpy().astype(np.float32))
            results.update(center=data['center'].detach().cpu().numpy().astype(np.float32))
            results.update(init_pose=data['init_pose'].detach().cpu().numpy().astype(np.float32))
            # save
            with open(f'final_pred_{i}.pkl', 'wb') as f:
                pickle.dump(results, f)
                
            loss, cur_loss_dict = loss_func.calcul_testloss(pred, data)
            loss_batch = loss.mean().detach() #/ batchsize
            # delete instance in cur_loss_dict
            if 'MPJPE_Instance' in cur_loss_dict.keys():
                cur_loss_dict.pop('MPJPE_Instance', None)
            if 'MPJPE_Instance_wrt_one' in cur_loss_dict.keys():
                cur_loss_dict.pop('MPJPE_Instance_wrt_one', None)
            print('batch: %d/%d, loss: %.6f ' %(i, len(loader), loss_batch), cur_loss_dict)
    return None


# def reconstruction_eval(model, loader, loss_func, device=torch.device('cpu')):
#     print('-' * 10 + 'model eval' + '-' * 10)
#     loss_all = 0.
#     model.model.eval()
#     output = {'pose':{}, 'shape':{}, 'trans':{}}
#     gt = {'pose':{}, 'shape':{}, 'trans':{}, 'gender':{}, 'valid':{}}
#     with torch.no_grad():
#         for i, data in tqdm(enumerate(loader), total=len(loader)):
#             # if i > 1:
#             #     break
#             batchsize = data['keypoints'].shape[0]
#             seq_id = data['seq_id']
#             frame_id = torch.cat(data['frame_id']).reshape(-1, batchsize)
#             frame_id = frame_id.detach().cpu().numpy().T

#             batch_size, frame_length, agent_num = data['keypoints'].shape[:3]

#             del data['seq_id']
#             del data['frame_id']
#             data = to_device(data, device)
#             data = extract_valid(data)

#             # forward
#             pred = model.model(data)

#             pred_pose = pred['pred_pose'].reshape(batch_size, frame_length, agent_num, -1)
#             pred_shape = pred['pred_shape'].reshape(batch_size, frame_length, agent_num, -1)
#             pred_trans = pred['pred_cam_t'].reshape(batch_size, frame_length, agent_num, -1)

#             pred_pose = pred_pose.detach().cpu().numpy()
#             pred_shape = pred_shape.detach().cpu().numpy()
#             pred_trans = pred_trans.detach().cpu().numpy()

#             gt_pose = data['pose'].reshape(batch_size, frame_length, agent_num, -1)
#             gt_shape = data['betas'].reshape(batch_size, frame_length, agent_num, -1)
#             gt_trans = data['gt_cam_t'].reshape(batch_size, frame_length, agent_num, -1)
#             gt_gender = data['gender'].reshape(batch_size, frame_length, agent_num)
#             valid = data['valid'].reshape(batch_size, frame_length, agent_num)

#             gt_pose = gt_pose.detach().cpu().numpy()
#             gt_shape = gt_shape.detach().cpu().numpy()
#             gt_trans = gt_trans.detach().cpu().numpy()
#             gt_gender = gt_gender.detach().cpu().numpy()
#             valid = valid.detach().cpu().numpy()

#             for batch in range(batchsize):
#                 s_id = str(int(seq_id[batch]))
#                 for f in range(frame_length):

#                     if s_id not in output['pose'].keys():
#                         output['pose'][s_id] = [pred_pose[batch][f]]
#                         output['shape'][s_id] = [pred_shape[batch][f]]
#                         output['trans'][s_id] = [pred_trans[batch][f]]

#                         gt['pose'][s_id] = [gt_pose[batch][f]]
#                         gt['shape'][s_id] = [gt_shape[batch][f]]
#                         gt['trans'][s_id] = [gt_trans[batch][f]]
#                         gt['gender'][s_id] = [gt_gender[batch][f]]
#                         gt['valid'][s_id] = [valid[batch][f]]
#                     else:
#                         output['pose'][s_id].append(pred_pose[batch][f])
#                         output['shape'][s_id].append(pred_shape[batch][f])
#                         output['trans'][s_id].append(pred_trans[batch][f])

#                         gt['pose'][s_id].append(gt_pose[batch][f])
#                         gt['shape'][s_id].append(gt_shape[batch][f])
#                         gt['trans'][s_id].append(gt_trans[batch][f])
#                         gt['gender'][s_id].append(gt_gender[batch][f])
#                         gt['valid'][s_id].append(valid[batch][f])
            
#             if False: #loss.max() > 100:
#                 results = {}
#                 results.update(imgs=data['imgname'])
#                 results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
#                 results.update(pred_pose=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
#                 results.update(pred_shape=pred['pred_shape'].detach().cpu().numpy().astype(np.float32))
#                 results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
#                 results.update(gt_pose=data['pose'].detach().cpu().numpy().astype(np.float32))
#                 results.update(gt_shape=data['betas'].detach().cpu().numpy().astype(np.float32))
#                 results.update(img_h=data['img_h'].detach().cpu().numpy().astype(np.float32))
#                 results.update(img_w=data['img_w'].detach().cpu().numpy().astype(np.float32))
#                 results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
#                 model.save_params(results, i, batchsize)


#             if False: #loss.max() > 100:
#                 results = {}
#                 results.update(imgs=data['imgname'])
#                 results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
#                 results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
#                 results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
#                 # results.update(MPJPE=loss.detach().cpu().numpy().astype(np.float32))
#                 if 'pred_verts' not in pred.keys():
#                     results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
#                     results.update(gt_joints=data['gt_joints'].detach().cpu().numpy().astype(np.float32))
#                     model.save_joint_results(results, i, batchsize)
#                 else:
#                     results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
#                     results.update(gt_verts=data['verts'].detach().cpu().numpy().astype(np.float32))
#                     model.save_results(results, i, batchsize, save_all = True)
#                     # results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
#                     # results.update(gt_joints=data['gt_joints'].detach().cpu().numpy().astype(np.float32))
#                     # model.save_joint_results(results, i, batchsize)
#         return output, gt

def process_main_model_output(data, pred):
    pose_6d = data['pose_6d']
    pred_pose6d = pred['pred_pose6d']
    pred_pose6d = pred_pose6d.reshape_as(pose_6d)
    pose_6d_model_output = torch.cat([pose_6d, pred_pose6d], dim=0) # [batchsize * 2, ...]
    pose_6d = torch.cat([pose_6d, pose_6d], dim=0)
    
    gt_cam_t = data['gt_cam_t']
    pred_cam_t = pred['pred_cam_t']
    pred_cam_t = pred_cam_t.reshape_as(gt_cam_t)
    gt_cam_t_model_output = torch.cat([gt_cam_t, pred_cam_t], dim=0) # [batchsize * 2, 3]
    gt_cam_t = torch.cat([gt_cam_t, gt_cam_t], dim=0)
    
    gt_joints = data['gt_joints'][...,:3]
    pred_joints = pred['pred_joints']
    pred_joints = pred_joints.reshape_as(gt_joints)
    gt_joints_model_output = torch.cat([gt_joints, pred_joints], dim=0) # [batchsize * 2, 17, 3]
    gt_joints = torch.cat([gt_joints, gt_joints], dim=0)
    
    gt_shape = data['betas']
    pred_shape = pred['pred_shape']
    pred_shape = pred_shape.reshape_as(gt_shape)
    gt_shape_model_output = torch.cat([gt_shape, pred_shape], dim=0) # [batchsize * 2, 10]
    gt_shape = torch.cat([gt_shape, gt_shape], dim=0)
    
    data['pose_6d'] = pose_6d
    data['gt_cam_t'] = gt_cam_t
    data['gt_joints'] = gt_joints
    data['betas'] = gt_shape
    
    for k in ['center', 'scale', 'img_h', 'img_w', 'focal_length', 'valid', 'has_3d', 'has_smpl', 'verts', 'contact_correspondences', 'pose', 'single_person']:
        tmp_data = data[k]
        tmp_data = torch.cat([tmp_data, tmp_data], dim=0)
        data[k] = tmp_data
        
    data['imgname'] = data['imgname'] * 2
    dshape = data['data_shape']
    data['data_shape'] = [dshape[0]*2, *dshape[1:]]
    data_model_output = {
        'pose_6d': pose_6d_model_output,
        'gt_cam_t': gt_cam_t_model_output,
        'gt_joints': gt_joints_model_output,
        'betas': gt_shape_model_output
    }
    
    return data, data_model_output
    
def prior_train(model, loss_func, train_loader, epoch, num_epoch, device=torch.device('cpu'), main_model=None):
    writer = SummaryWriter(model.output)
    print('-' * 10 + 'model training' + '-' * 10)
    len_data = len(train_loader)
    model.model.train(mode=True)
    if model.scheduler is not None:
        model.scheduler.step()

    train_loss = 0.
    for i, data in enumerate(train_loader):
        # if i > 1:
        #     break
        batchsize = data['keypoints'].shape[0]
        data = to_device(data, device)
        data = extract_valid(data)
        
        main_model_pred = main_model.model(data)
        
        data, data_model_pred = process_main_model_output(data, main_model_pred)

        # forward
        pred, commit_loss, perplexity = model.model(data_model_pred)

        # calculate loss
        loss, cur_loss_dict = loss_func.calcul_trainloss(pred, data)
        
        # add commit_loss to loss and cur_loss_dict
        loss += commit_loss
        cur_loss_dict['commit_loss'] = commit_loss
        cur_loss_dict['perplexity'] = perplexity

        debug = False
        if debug:
            results = {}
            results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(pred_pose=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
            results.update(pred_shape=pred['pred_shape'].detach().cpu().numpy().astype(np.float32))
            results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_pose=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_shape=pred['pred_shape'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
            model.save_generated_interaction(results, i, batchsize)

        debug = False
        if debug:
            results = {}
            results.update(imgs=data['imgname'])
            results.update(single_person=data['single_person'])
            results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
            if 'pred_verts' not in pred.keys():
                results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_joints=data['gt_joints'].detach().cpu().numpy().astype(np.float32))
                model.save_joint_results(results, i, batchsize)
            else:
                results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_verts=data['verts'].detach().cpu().numpy().astype(np.float32))
                model.save_results(results, i, batchsize)

        # backward
        model.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(parameters=model.model.parameters(), max_norm=100, norm_type=2)

        # optimize
        model.optimizer.step()
        if model.scheduler is not None:
            model.scheduler.batch_step()

        loss_batch = loss.detach()
        print('epoch: %d/%d, batch: %d/%d, loss: %.6f' %(epoch, num_epoch, i, len_data, loss_batch), cur_loss_dict)
        writer.add_scalar('Loss/train', loss_batch, epoch * len_data + i)
        for key, value in cur_loss_dict.items():
            writer.add_scalar(f'LossDict/train/{key}', value, epoch * len_data + i)
        train_loss += loss_batch
    writer.close()
    return train_loss/len_data

def prior_test(model, loss_func, loader, epoch, device=torch.device('cpu'), main_model=None):
    print('-' * 10 + 'model testing' + '-' * 10)
    loss_all = 0.
    loss_dict_all = {}
    model.model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            # if i > 10:
            #     break
            if i != 0:
                continue
            batchsize, frame_length, agent_num = data['keypoints'].shape[:3]
            data = to_device(data, device)
            data = extract_valid(data)
            
            main_model_pred = main_model.model(data)
        
            data, data_model_pred = process_main_model_output(data, main_model_pred)

            # forward
            pred, commit_loss, perplexity = model.model(data_model_pred)
            
            # seq_len = data['seq_len']
            # idx = data['idx']
            # last_window = (idx + frame_length + 3) >= seq_len
            
            # pred, data = extract_sliding_window(pred, data, [batchsize, frame_length, agent_num], window_size=3, last_window=last_window)

            # calculate loss
            loss, cur_loss_dict = loss_func.calcul_testloss(pred, data)
            
            if False: #loss.max() > 100:
                results = {}
                results.update(imgs=data['imgname'])
                results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_pose=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_shape=pred['pred_shape'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_pose=data['pose'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_shape=data['betas'].detach().cpu().numpy().astype(np.float32))
                results.update(img_h=data['img_h'].detach().cpu().numpy().astype(np.float32))
                results.update(img_w=data['img_w'].detach().cpu().numpy().astype(np.float32))
                results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
                model.save_params(results, i, batchsize)


            if False: # loss<130: #loss.max() > 100:
                results = {}
                results.update(valid=data['valid'])
                results.update(imgs=data['imgname'])
                results.update(single_person=data['single_person'])
                results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
                # if 'MPJPE_Instance' in cur_loss_dict.keys():
                #     MPJPE_Instance=cur_loss_dict['MPJPE_Instance']
                #     # save to a npy file
                #     np.save(f'mpjpe_{i}.npy', MPJPE_Instance.detach().cpu().numpy().astype(np.float32))
                # if 'Vel_Loss_Instance' in cur_loss_dict.keys():
                #     Vel_Loss_Instance=cur_loss_dict['Vel_Loss_Instance']
                #     # save to a npy file
                #     np.save(f'vel_loss_{i}.npy', Vel_Loss_Instance.detach().cpu().numpy().astype(np.float32))
                if 'pred_verts' not in pred.keys():
                    results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_joints=data['gt_joints'].detach().cpu().numpy().astype(np.float32))
                    model.save_joint_results(results, i, batchsize)
                else:
                    results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_verts=data['verts'].detach().cpu().numpy().astype(np.float32))
                    model.save_results(results, i, batchsize, True)
                    # results.update(pred_keypoints_2d = pred['pred_keypoints_2d'].detach().cpu().numpy().astype(np.float32))
                    # results.update(psudu_gt_keypoints_2d = data['pred_keypoints'].detach().cpu().numpy().astype(np.float32))
                    # results.update(gt_keypoints_2d = data['keypoints'].detach().cpu().numpy().astype(np.float32))
                    # results.update(center = data['center'].detach().cpu().numpy().astype(np.float32))
                    # model.save_2djoints(results, i, batchsize, False)

            loss_batch = loss.mean().detach() #/ batchsize
            print('batch: %d/%d, loss: %.6f ' %(i, len(loader), loss_batch), cur_loss_dict)
            loss_all += loss_batch
            for key, value in cur_loss_dict.items():
                if key not in loss_dict_all.keys():
                    loss_dict_all[key] = [value]
                else:
                    loss_dict_all[key].append(value)
        loss_all = loss_all / len(loader)
        # cal mean of loss_dict_all
        for key, value in loss_dict_all.items():
            if 'Instance' in key:
                continue
            loss_dict_all[key] = np.mean(value)
        return loss_all, loss_dict_all


def affordance_train(model, loss_func, train_loader, epoch, num_epoch, device=torch.device('cpu'), **kwargs):
    
    num_vertices = 431 + 1024
    mvm_mask = np.ones((num_vertices,))
    mvm_percent = 0.3
    pb = np.random.random_sample()
    masked_num = int(pb * mvm_percent * num_vertices) # at most x% of the vertices could be masked
    indices = np.random.choice(np.arange(num_vertices),replace=False,size=masked_num)
    mvm_mask[indices] = 0.0
    mvm_mask = torch.from_numpy(mvm_mask).float()

    writer = SummaryWriter(model.output)
    print('-' * 10 + 'model training' + '-' * 10)
    len_data = len(train_loader)
    model.model.train(mode=True)
    if model.scheduler is not None:
        model.scheduler.step()

    train_loss = 0.
    for i, data in enumerate(train_loader):
        # if i > 10:
        #     break
        batchsize = data['features'].shape[0]
        data = to_device(data, device)
        data = extract_valid(data)
        data['mvm_mask'] = mvm_mask.to(device)

        # forward
        pred = model.model(data)

        # calculate loss
        loss, cur_loss_dict = loss_func.calcul_trainloss(pred, data)

        if i<0: # loss<130: #loss.max() > 100:
            results = {}
            results.update(valid=data['valid'])
            results.update(imgs=data['imgname'])
            results.update(single_person=data['single_person'])
            results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_verts=data['verts'].detach().cpu().numpy().astype(np.float32))
            results.update(pred_verts_contact=pred['afford_full'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_verts_contact=data['contact_correspondences'].detach().cpu().numpy().astype(np.float32))
            model.save_contact_results(results, i, batchsize, True)

        # backward
        model.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(parameters=model.model.parameters(), max_norm=100, norm_type=2)

        # optimize
        model.optimizer.step()
        if model.scheduler is not None:
            model.scheduler.batch_step()

        loss_batch = loss.detach()
        print('epoch: %d/%d, batch: %d/%d, loss: %.6f' %(epoch, num_epoch, i, len_data, loss_batch), cur_loss_dict)
        writer.add_scalar('Loss/train', loss_batch, epoch * len_data + i)
        for key, value in cur_loss_dict.items():
            writer.add_scalar(f'LossDict/train/{key}', value, epoch * len_data + i)
        train_loss += loss_batch
    writer.close()
    return train_loss/len_data

def affordance_test(model, loss_func, loader, epoch, device=torch.device('cpu'), **kwargs):
    print('-' * 10 + 'model testing' + '-' * 10)
    loss_all = 0.
    loss_dict_all = {}
    model.model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            # if i != 0 and i !=12:
            #     continue
            # if i != 0:
            #     continue
            batchsize, frame_length, agent_num = data['keypoints'].shape[:3]
            data = to_device(data, device)
            data = extract_valid(data)

            # forward
            pred = model.model(data)
            
            # seq_len = data['seq_len']
            # idx = data['idx']
            # last_window = (idx + frame_length + 3) >= seq_len
            
            # pred, data = extract_sliding_window(pred, data, [batchsize, frame_length, agent_num], window_size=3, last_window=last_window)

            # calculate loss
            loss, cur_loss_dict = loss_func.calcul_testloss(pred, data)

            if i<1: # loss<130: #loss.max() > 100:
                results = {}
                results.update(valid=data['valid'])
                results.update(imgs=data['imgname'])
                results.update(single_person=data['single_person'])
                results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_verts=data['verts'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_verts_contact=pred['afford_full'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_verts_contact=data['contact_correspondences'].detach().cpu().numpy().astype(np.float32))
                model.save_contact_results(results, i, batchsize, True)

            loss_batch = loss.mean().detach()
            print('batch: %d/%d, loss: %.6f ' %(i, len(loader), loss_batch), cur_loss_dict)
            loss_all += loss_batch
            for key, value in cur_loss_dict.items():
                if key not in loss_dict_all.keys():
                    loss_dict_all[key] = [value]
                else:
                    loss_dict_all[key].append(value)
        loss_all = loss_all / len(loader)
        # cal mean of loss_dict_all
        for key, value in loss_dict_all.items():
            loss_dict_all[key] = np.mean(value)
        return loss_all, loss_dict_all

def reconstruction_j3d_train(model, loss_func, train_loader, epoch, num_epoch, device=torch.device('cpu'), main_model=None):
    writer = SummaryWriter(model.output)
    print('-' * 10 + 'model training' + '-' * 10)
    len_data = len(train_loader)
    model.model.train(mode=True)
    if model.scheduler is not None:
        model.scheduler.step()

    train_loss = 0.
    for i, data in enumerate(train_loader):
        # if i > 0:
        #     break
        batchsize = data['keypoints'].shape[0]
        data = to_device(data, device)
        data = extract_valid(data)
        
        main_model_pred = main_model.model(data)
        
        data['pred_joints'] = main_model_pred['pred_joints']
        data['pred_keypoints_2d'] = main_model_pred['pred_keypoints_2d']
        data['pred_cam_t'] = main_model_pred['pred_cam_t']

        # forward
        pred = model.model(data)

        # calculate loss
        loss, cur_loss_dict = loss_func.calcul_trainloss(pred, data)

        debug = False
        if debug:
            results = {}
            results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(pred_pose=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
            results.update(pred_shape=pred['pred_shape'].detach().cpu().numpy().astype(np.float32))
            results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_pose=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_shape=pred['pred_shape'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
            model.save_generated_interaction(results, i, batchsize)

        debug = False
        if debug:
            results = {}
            results.update(imgs=data['imgname'])
            results.update(single_person=data['single_person'])
            results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
            results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
            if 'pred_verts' not in pred.keys():
                results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_joints=data['gt_joints'].detach().cpu().numpy().astype(np.float32))
                model.save_joint_results(results, i, batchsize)
            else:
                results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_verts=data['verts'].detach().cpu().numpy().astype(np.float32))
                model.save_results(results, i, batchsize)

        # backward
        model.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(parameters=model.model.parameters(), max_norm=100, norm_type=2)

        # optimize
        model.optimizer.step()
        if model.scheduler is not None:
            model.scheduler.batch_step()

        loss_batch = loss.detach()
        print('epoch: %d/%d, batch: %d/%d, loss: %.6f' %(epoch, num_epoch, i, len_data, loss_batch), cur_loss_dict)
        writer.add_scalar('Loss/train', loss_batch, epoch * len_data + i)
        for key, value in cur_loss_dict.items():
            writer.add_scalar(f'LossDict/train/{key}', value, epoch * len_data + i)
        train_loss += loss_batch
    writer.close()
    return train_loss/len_data

def reconstruction_j3d_test(model, loss_func, loader, epoch, device=torch.device('cpu'), main_model=None):
    print('-' * 10 + 'model testing' + '-' * 10)
    loss_all = 0.
    valid_sum = 0.
    loss_dict_all = {}
    model.model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            # if i > 10:
            #     break
            # if i != 0:
            #     continue
            batchsize, frame_length, agent_num = data['keypoints'].shape[:3]
            data = to_device(data, device)
            data = extract_valid(data)
            
            main_model_pred = main_model.model(data)
        
            data['pred_joints'] = main_model_pred['pred_joints']
            data['pred_keypoints_2d'] = main_model_pred['pred_keypoints_2d']
            data['pred_cam_t'] = main_model_pred['pred_cam_t']

            # forward
            pred = model.model(data)

            # calculate loss
            loss, cur_loss_dict = loss_func.calcul_testloss(pred, data)
            
            if i==0: # loss<130: #loss.max() > 100:
                results = {}
                results.update(valid=data['valid'])
                results.update(imgs=data['imgname'])
                results.update(single_person=data['single_person'])
                results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))

                # results.update(pred_keypoints_2d = pred['pred_keypoints_2d'].detach().cpu().numpy().astype(np.float32))
                # results.update(psudu_gt_keypoints_2d = data['pred_keypoints_2d'].detach().cpu().numpy().astype(np.float32))
                # results.update(gt_keypoints_2d = data['keypoints'].detach().cpu().numpy().astype(np.float32))
                # results.update(center = data['center'].detach().cpu().numpy().astype(np.float32))
                # model.save_2djoints(results, i, batchsize, True)

                pred_trans = torch.zeros_like(data['gt_cam_t'])
                results.update(pred_trans=pred_trans.detach().cpu().numpy().astype(np.float32))
                results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_joints=pred['pred_joints'].reshape_as(data['gt_joints'][...,:3]).detach().cpu().numpy().astype(np.float32))
                results.update(gt_joints=data['gt_joints'].detach().cpu().numpy().astype(np.float32))
                model.save_joint_results(results, i, batchsize, False)

            loss_batch = loss.mean().detach() #/ batchsize
            if 'MPJPE_Instance' in cur_loss_dict.keys():
                cur_loss_dict.pop('MPJPE_Instance', None)
            if 'MPJPE_Instance_wrt_one' in cur_loss_dict.keys():
                cur_loss_dict.pop('MPJPE_Instance_wrt_one', None)
            print('batch: %d/%d, loss: %.6f ' %(i, len(loader), loss_batch), cur_loss_dict)
            loss_all += loss_batch
            valid_sum += data['valid'].sum().detach().cpu().numpy()
            for key, value in cur_loss_dict.items():
                if key not in loss_dict_all.keys():
                    loss_dict_all[key] = [value]
                else:
                    loss_dict_all[key].append(value)
        loss_all = loss_all / valid_sum
        # cal mean of loss_dict_all
        for key, value in loss_dict_all.items():
            if 'Instance' in key:
                continue
            loss_dict_all[key] = np.mean(value) / valid_sum
        return loss_all, loss_dict_all
try:
    from qwen_vl_utils import process_vision_info
except Exception as e:
    print(e)

def gen_text(loader, device, processor, model):
    print('-' * 10 + 'Generate Text Description' + '-' * 10)
    len_data = len(loader)

    for i, data in enumerate(loader):
        # if i > 10:
        #     break
        batchsize, frame_length, agent_num = data['features'].shape[:3]
        data = to_device(data, device)
        data = extract_valid(data)
        img_name = data['imgname'] # [batchsize*frame_length]
        messages = []
        for b in range(batchsize):
            img_path_list = img_name[b*frame_length:(b+1)*frame_length]
            message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": [
                                # f"file:///root/autodl-tmp/Hi4D/images/{seq_name}/{camera_name}/000{str(i).zfill(3)}.jpg" for i in range(start_frame, end_frame+1)
                                f"file:///{img_path}" for img_path in img_path_list
                            ],
                        },
                        {"type": "text", "text": 
                        "Given the image sequence of two human interaction, generate 0 to 3 joint-joint contact pair(s) according to the following background information, rules, and examples. Joint-joint contact pair should exactly reflect the human interaction shown in the image sequence. \
                        [start of background Information] \
                        JOINTS list: ['pelvis', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']. \
                        [end of background Information] \
                        [start of rules] \
                        1.Each joint-joint pair should be formatted into {JOINT, JOINT, TIME-STEP, TIME-STEP}. JOINT should be replaced by one of JOINT in the JOINT list in the background information. IMPORTANT: The first JOINT belongs to person 1, and the second JOINT belongs to person 2. Each joint-joint pair represents a contact of a joint of person 1 and a joint of person 2. The first TIME-STEP is the start frame number of contact, and the second TIME-STEP is the end frame number of contact. IMPORTANT: JOINT must be select from joints list in the background information, do not use other words. IMPORTANT: TIME-STEP should accurately reflect the time when the joints in the image sequence came into contact. \
                        2.Use one sentence to describe what action person 1 do and one sentence to describe what action person 2 do according to the image sequence. IMPORTANT: the sentence starts from 'text 1:' describing the action of person 1 from the perspective of person 1 and the sentence starts from 'text 2:' describing the action of person 2 from the perspective of person 2. Sentences should NOT contain words like 'person 1' or 'person 2', use 'a person' to refer to himself in the sentence and 'others' to refer to others. IMPORTANT: the order of person 1 and person 2 should be the same in different joint-joint contact pair of the same image sequence.\
                        3.IMPORTANT: Do NOT add explanations for the joint-joint contact pair. \
                        4.Do not generate joint-joint contact pair that are not mentioned in the sentences. \
                        5.Your respose should be in the format provided in the examples, use [start of sentences] and [end of sentences] to mark the start and end of sentences, use [Start of joint-joint contact pair(s)] and [End of joint-joint contact pair(s)] to mark the start and end of joint-joint contact pair(s).\
                        6.IMPORTANT: If there is no contact between joints, do not generate any joint-joint contact pair. \
                        [end of rules] \
                        [start of example that contains joint-joint contact pair(s)] \
                        [Start of sentences] \
                        Text 1: a person dance with others holding his left hand with the other's right hand, puting his right hand on the other's waist, and his shoulder being touched. \
                        Text 2: a person dance with other holding her right hand with the other's left hand, with her waist being embraced, placing her left hand on the other's shoulder. \
                        [End of sentences] \
                        [Start of joint-joint contact pair(s)] \
                        {left_wrist, right_wrist, 0, 15} \
                        {right_wrist, left_hip, 5, 10} \
                        {right_shoulder, left_wrist, 9, 15} \
                        [End of joint-joint contact pair(s)] \
                        [end of example that contains joint-joint contact pair(s)] \
                        \
                        [start of example that does not contain joint-joint contact pair(s)] \
                        [Start of sentences] \
                        Text 1: A person stands facing another person without any contact between them. \
                        Text 2: A person stands facing another person without any contact between them. \
                        [End of sentences] \
                        [Start of joint-joint contact pair(s)] \
                        [End of joint-joint contact pair(s)] \
                        [end of example that does not contain joint-joint contact pair(s)]"
                        },
                    ],
                }
            ]
            messages.append(message)

        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side='left',
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Batch Inference
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # save the generated text
        for j, output_text in enumerate(output_texts):
            img_path_list = img_name[j*frame_length:(j+1)*frame_length]
            img_path = img_path_list[0]
            seq_name = img_path.split('/')[-3]
            cam_name = img_path.split('/')[-2]
            frame_start = img_path.split('/')[-1].split('.')[0]
            img_path = img_path_list[-1]
            frame_end = img_path.split('/')[-1].split('.')[0]
            print(f"seq_name: {seq_name}, cam_name: {cam_name}, frame_start: {frame_start}, frame_end: {frame_end}")
            print(output_text)
            path = '/root/autodl-tmp/Hi4D/text'
            with open(f'{path}/{seq_name}_{cam_name}_{frame_start}_{frame_end}.txt', 'w') as f:
                f.write(output_text)
        print('batch: %d/%d,' %(i, len(loader)))

        
    return 0

import json
JOINT_NAMES = ["Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist", "LHip", "RHip", "LKnee", "Rknee", "LAnkle", "RAnkle", "Head", "Neck", "Hip", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel"]
# {0,  "Nose"},
# {1,  "LEye"},
# {2,  "REye"},
# {3,  "LEar"},
# {4,  "REar"},
# {5,  "LShoulder"},
# {6,  "RShoulder"},
# {7,  "LElbow"},
# {8,  "RElbow"},
# {9,  "LWrist"},
# {10, "RWrist"},
# {11, "LHip"},
# {12, "RHip"},
# {13, "LKnee"},
# {14, "Rknee"},
# {15, "LAnkle"},
# {16, "RAnkle"},
# {17,  "Head"},
# {18,  "Neck"},
# {19,  "Hip"},
# {20, "LBigToe"},
# {21, "RBigToe"},
# {22, "LSmallToe"},
# {23, "RSmallToe"},
# {24, "LHeel"},
# {25, "RHeel"},
# {1,  "LEye"},
# {2,  "REye"},
# {3,  "LEar"},
# {4,  "REar"},
# {5,  "LShoulder"},
# {6,  "RShoulder"},
# {7,  "LElbow"},
# {8,  "RElbow"},
# {9,  "LWrist"},
# {10, "RWrist"},
# {11, "LHip"},
# {12, "RHip"},
# {13, "LKnee"},
# {14, "Rknee"},
# {15, "LAnkle"},
# {16, "RAnkle"},
# {17,  "Head"},
# {18,  "Neck"},
# {19,  "Hip"},
# {20, "LBigToe"},
# {21, "RBigToe"},
# {22, "LSmallToe"},
# {23, "RSmallToe"},
# {24, "LHeel"},
# {25, "RHeel"},

def merge_contact_pairs(contact_pairs):
    '''
    1. merge same contact pairs
    2. merge continuous contact pairs
    3. find min max dist for each contact pair
    '''
    contact_pairs = sorted(contact_pairs, key=lambda x: x[2])
    contact_pairs = sorted(contact_pairs, key=lambda x: x[0])
    contact_pairs = sorted(contact_pairs, key=lambda x: x[1])
    merged_contact_pairs = []
    for cp in contact_pairs:
        if len(merged_contact_pairs) == 0:
            merged_contact_pairs.append(cp + [cp[4], cp[4]])  # Add min and max distance
        else:
            last_cp = merged_contact_pairs[-1]
            if cp[0] == last_cp[0] and cp[1] == last_cp[1] and cp[2] == last_cp[3] + 1:
                last_cp[3] = cp[3]
                last_cp[4] = min(last_cp[4], cp[4])  # Update min distance
                last_cp[5] = max(last_cp[5], cp[4])  # Update max distance
            else:
                merged_contact_pairs.append(cp + [cp[4], cp[4]])  # Add min and max distance
    return merged_contact_pairs

def contact_pairs2text(contact_pairs):
    '''
    convert contact pairs to text
    '''
    text = ''
    head = '[Start of joint-joint contact pair(s)] \n'
    tail = '[End of joint-joint contact pair(s)] \n'
    text += head
    for cp in contact_pairs:
        text += '{' + JOINT_NAMES[cp[0]] + ', ' + JOINT_NAMES[cp[1]] + ', ' + str(cp[2]) + ', ' + str(cp[3]) + ',' + str(cp[4]) + ',' + str(cp[5]) + '}\n'
    text += tail
    return text

def gen_contact_pair(loader, device, contact_threshold = 0.1, json_path = 'contact_pairs.json'):
    print('-' * 10 + 'Generate Contact Pairs' + '-' * 10)
    len_data = len(loader)
    json_datas = []
    for i, data in enumerate(loader):
        # if i > 3:
        #     break
        batchsize, frame_length, agent_num = data['features'].shape[:3]
        # data = to_device(data, device)
        data = extract_valid(data)
        img_name = data['imgname'] # [batchsize*frame_length]
        gt_joints = data['gt_joints'].reshape(batchsize, frame_length, agent_num, 26, 4)[...,:3]
        gt_trans = data['gt_cam_t'].reshape(batchsize, frame_length, agent_num, 3)
        gt_g_joints = gt_joints + gt_trans[:, :, :, None, :]
        joitns_a = gt_g_joints[:, :, 0]
        joitns_b = gt_g_joints[:, :, 1]
        cdist_joints = torch.cdist(joitns_a, joitns_b, p=2).cpu().numpy()
        # round to 2 decimal
        cdist_joints = np.round(cdist_joints, 2)

        
        for b in range(batchsize):
            contact_pairs = []
            for f in range(frame_length):
                for j1 in range(26):
                    for j2 in range(26):
                        if cdist_joints[b, f, j1, j2] < contact_threshold:
                            contact_pairs.append([j1, j2, f, f, cdist_joints[b, f, j1, j2]])
            merged_contact_pair = merge_contact_pairs(contact_pairs)
            contact_pair_msg = contact_pairs2text(merged_contact_pair)
            print(img_name[b*frame_length], 'to', img_name[(b+1)*frame_length-1])
            print(contact_pair_msg)
            
            # save as json
            # json example
            # [
            # {
            #     "id": "000000033471",
            #     "image": ["000000033471.jpg", "000000033472.jpg"],
            #     "conversations": [
            #     {
            #         "from": "human",
            #         "value": "<image>\n<image>\nIs the perspective of the camera differnt?"
            #     },
            #     {
            #         "from": "gpt",
            #         "value": "Yes, It the perspective of the camera is different."
            #     }
            #     ]
            # }
            # ]

            # 16 img path
            img_path = img_name[b*frame_length:b*frame_length+16]
            json_data = {
                "id": i*batchsize+b,
                "image": img_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\n"*16 +
                        "Given the image sequence of two human interaction, generate 0 to 3 joint-joint contact pair(s) according to the following background information, rules, and examples. Joint-joint contact pair should exactly reflect the human interaction shown in the image sequence. \
                        [start of background Information] \
                        JOINTS list: ['Nose', 'LEye', 'REye', 'LEar', 'REar', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist', 'LHip', 'RHip', 'LKnee', 'Rknee', 'LAnkle', 'RAnkle', 'Head', 'Neck', 'Hip', 'LBigToe', 'RBigToe', 'LSmallToe', 'RSmallToe', 'LHeel', 'RHeel']. \
                        [end of background Information] \
                        1.Each joint-joint pair should be formatted into {JOINT, JOINT, TIME-STEP, TIME-STEP}. JOINT should be replaced by one of JOINT in the JOINT list in the background information. IMPORTANT: The first JOINT belongs to person 1, and the second JOINT belongs to person 2. Each joint-joint pair represents a contact of a joint of person 1 and a joint of person 2. The first TIME-STEP is the start frame number of contact, and the second TIME-STEP is the end frame number of contact. IMPORTANT: JOINT must be select from joints list in the background information, do not use other words. IMPORTANT: TIME-STEP should accurately reflect the time when the joints in the image sequence came into contact. \
                        2.IMPORTANT: Do NOT add explanations for the joint-joint contact pair. \
                        3.Your respose should be in the format provided in the examples, use [Start of joint-joint contact pair(s)] and [End of joint-joint contact pair(s)] to mark the start and end of joint-joint contact pair(s).\
                        4.IMPORTANT: If there is no contact between joints, do not generate any joint-joint contact pair. \
                        [end of rules] \
                        [start of an example] \
                        Instruction: dance \
                        [Start of sentences] \
                        Text 1: a person dance with others holding his left hand with the other's right hand, puting his right hand on the other's waist, and his shoulder being touched. \
                        Text 2: a person dance with other holding her right hand with the other's left hand, with her waist being embraced, placing her left hand on the other's shoulder. \
                        [End of sentences] \
                        [Start of joint-joint contact pair(s)] \
                        {left_wrist, right_wrist, 0, 15} \
                        {right_wrist, left_hip, 5, 10} \
                        {right_shoulder, left_wrist, 9, 15} \
                        [End of joint-joint contact pair(s)] \
                        [end of an example]"
                    },
                    {
                        "from": "gpt",
                        "value": contact_pair_msg
                    }
                ]
            }
            json_datas.append(json_data)
                
    with open(json_path, 'a') as f:
        json.dump(json_datas, f)
        
    return 0