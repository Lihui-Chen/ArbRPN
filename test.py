import argparse
import time
import os
import options.options as option
from utils import util
from solvers import create_solver
from data import create_dataloader
from data import create_dataset
import numpy as np


def main():
    args = option.add_args()
    opt = option.parse(args)
    opt = option.dict_to_nonedict(opt)

    # initial configure
    scale = opt['scale']
    degrad = opt['degradation']
    network_opt = opt['networks']
    model_name = network_opt['which_model'].upper()
    if opt['self_ensemble']:
        model_name += 'plus'

    # create test dataloader
    bm_names = []
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        test_loaders.append(test_loader)
        print('===> Test Dataset: [%s]   Number of images: [%d]' % (
            test_set.name(), len(test_set)))
        bm_names.append(test_set.name())

    # create solver (and load model)
    solver = create_solver(opt)
    # Test phase
    print('===> Start Test')
    print("==================================================")
    print("Method: %s || Scale: %d || Degradation: %s" %(model_name, scale, degrad))

    for bm, test_loader in zip(bm_names, test_loaders):
        print("Test set : [%s]" % bm)

        total_time = []

        need_HR = False if test_loader.dataset.__class__.__name__.find(
            'LRHR') < 0 else True

        if need_HR:
            save_img_path = os.path.join(
                './results/SR/', model_name, bm, "x%d" % scale)
        else:
            save_img_path = os.path.join(
                './results/SR/', model_name, bm,  "x%d" % scale)

        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)

        for iter, batch in enumerate(test_loader):
            solver.feed_data(batch, need_HR=need_HR)

            # calculate forward time
            t0 = time.time()
            solver.test()
            t1 = time.time()
            total_time.append((t1 - t0))
            visuals = solver.get_current_visual(need_HR=need_HR)
            # calculate PSNR/SSIM metrics on Python
            if need_HR:
                tmpMetricsDict = util.pan_calc_metrics_rr(visuals['SR'], visuals['HR'], opt['scale'], opt['img_range'])
                if iter == 0: metrics_list = {tKey:[] for tKey in tmpMetricsDict.keys()}
                for tKey, tValue in tmpMetricsDict.items():
                    metrics_list[tKey].append(tValue)
                print("[%d/%d] %s %s | Time: %.4f(s)."
                        %(iter + 1, len(test_loader), os.path.basename(batch['LR_path'][0]),
                        ''.join(['| %s: %.4f '%(t_key, t_value) for t_key, t_value in tmpMetricsDict.items()]),
                        (t1 - t0)))
            else:
                print("[%d/%d] %s || Timer: %.4f sec ."
                        % (iter + 1, len(test_loader),os.path.basename(batch['LR_path'][0]), (t1 - t0)))
            if opt['save_image']:
                name = ('x{}_' + model_name + '_').format(scale) + os.path.basename(batch['LR_path'][0])
                np.save(os.path.join(save_img_path, name), visuals['SR'])

        if need_HR:
            print("---- Average Metrics /Speed(s) for [%s] ----" % bm)
            metrics_keys = metrics_list.keys()
            metrics_values = ['%.4f'%(sum(metrics_list[t_key])/len(metrics_list[t_key])) for t_key in metrics_keys]
            print("     | Method | %s  |  Time  | \n     | ------ %s| ------ |  \n     | %s | %s | %.4f |"
                    % ('   |  '.join(metrics_keys), ''.join(['| ------ ' for _ in metrics_keys]),
                    network_opt['which_model'].upper()[:6],' | '.join(metrics_values), 
                    sum(total_time)/len(total_time)))
        else:
            print("---- Average Speed(s) for [%s] is %.4f sec ----"
                    % (bm, sum(total_time) / len(total_time)))

    print("==================================================")
    print("===> Finished !")


if __name__ == '__main__':
    main()
