'''
 @FileName    : train.py
 @EditTime    : 2024-04-01 16:05:48
 @Author      : Buzhen Huang
 @Email       : buzhenhuang@outlook.com
 @Description : 
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import DataLoader
from cmd_parser import parse_config
from utils.module_utils import seed_worker, set_seed
from modules import init, LossLoader, ModelLoader, DatasetLoader

###########Load config file in debug mode#########
# import sys
# sys.argv = ['','--config=cfg_files/config.yaml'] # interVAE config


def main(**args):
    seed = 42
    g = set_seed(seed)

    # Global setting
    dtype = torch.float32
    batchsize = args.get('batchsize')
    num_epoch = args.get('epoch')
    workers = args.get('worker')
    device = torch.device(index=args.get('gpu_index'), type='cuda')
    mode = args.get('mode')
    task = args.get('task')

    # Initialize project setting, e.g., create output folder, load SMPL model
    out_dir, logger, smpl = init(dtype=dtype, **args)

    # Load loss function
    loss = LossLoader(smpl, device=device, **args)

    # Load model
    model = ModelLoader(dtype=dtype, device=device, out_dir=out_dir, **args)
    
    main_model = None
    
    if task == 'prior' or task == 'reconstruction_j3d':
        main_model_dir = args.get('base_model')
        main_model = ModelLoader(dtype=dtype, device=device, out_dir=out_dir, model = 'interhuman_diffusion_phys', pretrain = True, pretrain_dir=main_model_dir, 
                                    lr=0.0, optimizer='adam', scheduler='none', scheduler_param=None, weight_decay=0.0,)
        main_model.model.requires_grad_(False)
        main_model.model.eval()

    # create data loader
    dataset = DatasetLoader(dtype=dtype, smpl=smpl, **args)
    if mode == 'train':
        train_dataset = dataset.load_trainset()
        train_loader = DataLoader(
            train_dataset,
            batch_size=batchsize, shuffle=True,
            num_workers=workers, pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        if args.get('use_sch'):
            model.load_scheduler(train_dataset.cumulative_sizes[-1])
            
    test_dataset = dataset.load_testset()
    test_loader = DataLoader(
        test_dataset,
        batch_size=batchsize, shuffle=False,
        num_workers=workers, pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Load handle function with the task name
    exec('from process import %s_train' %task)
    exec('from process import %s_test' %task)

    for epoch in range(num_epoch):
        # training mode
        if mode == 'train':
            training_loss = eval('%s_train' %task)(model, loss, train_loader, epoch, num_epoch, device=device, main_model = main_model)
            testing_loss, _ = eval('%s_test' %task)(model, loss, test_loader, epoch, device=device, main_model = main_model)
            lr = model.optimizer.state_dict()['param_groups'][0]['lr']
            logger.append([int(epoch + 1), lr, training_loss, testing_loss])
            # save trained model
            model.save_best_model(testing_loss, epoch, task)
            model.save_last_model()
            
        if epoch % 10 == 0 and mode == 'train':
            model.save_model(epoch, task)

        # testing mode
        elif epoch == 0 and mode == 'test':
            training_loss = -1.
            testing_loss, loss_dict = eval('%s_test' %task)(model, loss, test_loader, epoch, device=device, main_model = main_model)
            # testing_loss = float(testing_loss)  # Ensure testing_loss is a float
            print("TESTING RESULT")
            print(loss_dict)
            # write testing result to log file
            logger.write_test_result(loss_dict)

    logger.close()


if __name__ == "__main__":
    args = parse_config()
    main(**args)




