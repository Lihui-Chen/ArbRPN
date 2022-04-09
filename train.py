import random
# from tqdm import tqdm
import time
# from torch.utils.tensorboard import SummaryWriter
import torch
import options.options as option
from utils import util
from solvers import create_solver
from data import create_dataloader, data_prefetcher
from data import create_dataset
import os
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
import data.my_collate_fn as my_collate

def pytorch_seed(seed=0):
    print("===> Random Seed: [%d]" %seed)
    seed=int(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def main():
    args = option.add_args()
    opt = option.parse(args)


    # random seed
    seed = opt['solver']['manual_seed']
    if seed is None: seed = random.randint(1, 10000)
    pytorch_seed(seed)

    # create train and val dataloader
    train_loader_list = []
    bm_names = []
    collate_fn = None
    if opt['mask_training'] is not None:
        collate_fn = opt['mask_training']
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if 'train' in phase:
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt, collate_fn) #todo:
            train_loader_list.append(train_loader)
            print('===> Train Dataset: %s  Number of images: [%d]' % (train_set.name(), len(train_set)))
            if train_loader is None: raise ValueError("[Error] The training data does not exist")
            bm_names.append(train_set.name())
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print('===> Val Dataset: %s  Number of images: [%d]' % (val_set.name(), len(val_set)))

        else:
            raise NotImplementedError("[Error] Dataset phase [%s] in *.json is not recognized." % phase)

    solver = create_solver(opt)

    scale = opt['scale']
    model_name = opt['networks']['which_model'].upper()

    print('===> Start Train')
    print("==================================================")

    solver_log = solver.get_current_log()

    NUM_EPOCH = int(opt['solver']['num_epochs'])
    start_epoch = solver_log['epoch']

    print("Method: %s || Scale: %d || Epoch Range: (%d ~ %d)"%(model_name, scale, start_epoch, NUM_EPOCH))
    
    for epoch in range(start_epoch, NUM_EPOCH + 1):
        print('\n==============> Start Train [Train Set: %s; Val Set: %s]<=============='%(train_set.name(), val_set.name()))
        print('[Train]: [%d/%d] || loss_type: %s || Learning Rate: %f'%(epoch,
                                                                      NUM_EPOCH, opt['solver']['loss_type'],
                                                                      solver.get_current_learning_rate()))
        # Initialization
        solver_log['epoch'] = epoch

        # Train model
        train_loss_list = []
        total_len = [len(in_set) for in_set in train_loader_list]
        total_len = sum(total_len)

        start_time=time.time()
        for bm, train_loader in zip(bm_names, train_loader_list):
            print('--------------------> Sub-Train Set: %s<--------------------'%(bm))
            for iter, (batch, batch_len, mask) in enumerate(train_loader):
                solver.feed_data(batch)
                iter_loss = solver.train_step(mask)
                batch_size = batch['LR'].size(0)
                train_loss_list.append(iter_loss*batch_size)
        end_time = time.time()
        
        solver_log['records']['train_loss'].append(sum(train_loss_list)/len(train_set))
        solver_log['records']['lr'].append(solver.get_current_learning_rate())
        print('[Epoch]: [%d/%d] || No.Iter: %d || Avg Train Loss: %.6f || Time: %.1f' % (epoch,
                                                    NUM_EPOCH, total_len,
                                                    sum(train_loss_list)/len(train_set), end_time-start_time))

        # print('===> Validating...',)
    
        val_loss_list = []

        for iter, batch in enumerate(val_loader):
            solver.feed_data(batch)
            iter_loss = solver.test()
            val_loss_list.append(iter_loss)

            # calculate evaluation metrics
            visuals = solver.get_current_visual()
            tmpMetricsDict = util.pan_calc_metrics_all(visuals, scale, opt['img_range'], FR=False)
            if iter==0 and epoch ==start_epoch: metrics_list = {tKey:[] for tKey in tmpMetricsDict.keys()}
            for tKey, tValue in tmpMetricsDict.items():
                metrics_list[tKey].append(tValue)

        solver_log['records']['val_loss'].append(sum(val_loss_list)/len(val_loss_list))
        for key, value in metrics_list.items():
            value = sum(value)/len(value)
            solver_log['records'][key].append(value)

        # record the best epoch
        epoch_is_best = False
        if solver_log['best_pred'] < (sum(metrics_list['Q2n'])/len(metrics_list['Q2n'])):
            solver_log['best_pred'] = (sum(metrics_list['Q2n'])/len(metrics_list['Q2n']))
            epoch_is_best = True
            solver_log['best_epoch'] = epoch

        print("[ val ]: %s | Time: %.4f(s) \n         Best Epoch [%d] | Q2n: %.4f "
                        %(''.join(['| %s: %.4f '%(t_key, sum(t_value)/len(t_value)) for t_key, t_value in metrics_list.items()]),
                        (end_time-start_time), solver_log['best_epoch'], solver_log['best_pred']))

        solver.set_current_log(solver_log)
        solver.save_checkpoint(epoch, epoch_is_best)
        solver.save_current_log()

        # update lr
        solver.update_learning_rate(epoch)

    # writer.close()
    print('===> Finished !')


if __name__ == '__main__':
    main()