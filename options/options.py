import os
from collections import OrderedDict
from datetime import datetime
import yaml
import torch
from yaml.events import NodeEvent
from utils import util
import shutil
import argparse

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper



def add_args():
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    parser.add_argument('-trained_model', type=str, default=None)
    # network-setting
    parser.add_argument('-net_arch', type=str, default=None)
    # parser.add_argument('-num_layers', type=int, default=None)
    parser.add_argument('-hid_dim', type=int, default=None)
    parser.add_argument('-input_dim', type=int, default=None)
    parser.add_argument('-num_resblocks', type=int, default=None)
    parser.add_argument('-num_cycle', type=int, default=None)
    parser.add_argument('-in_channels', type=int, default=None)
    # dataset-setting
    parser.add_argument('-lrpath', type=str, default=None)
    parser.add_argument('-panpath', type=str, default=None)
    parser.add_argument('-datamode', type=str, default=None)
    # optimizer-setting
    parser.add_argument('-loss', type=str, default=None)
    parser.add_argument('-lr_step', nargs='+', type=int, default=None)
    parser.add_argument('-lr_scheme', type=str, default=None)
    parser.add_argument('-lr', type=float, default=None)
    # log-setting
    parser.add_argument('-save_name', type=str, default=None)
    return parser.parse_args()


def parse(args):
    Loader, Dumper = OrderedYaml()
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)

    opt['timestamp'] = get_timestamp()

    if getattr(args, 'trained_model') is not None:
        opt['solver']['pretrained_path'] = args.trained_model

    # for network-setting
    opt['networks']['scale'] = opt['scale']
    network_opt = opt['networks']
    if args.hid_dim:
        network_opt['hid_dim'] = args.hid_dim
    if args.input_dim:
        network_opt['input_dim'] = args.input_dim
    if args.net_arch:
        network_opt['which_model'] = args.net_arch
    if args.num_resblocks:
        network_opt['num_resblocks'] = args.num_resblocks
    if args.num_cycle:
        network_opt['num_cycle'] = args.num_cycle
    if args.in_channels:
        network_opt['in_channels'] = args.in_channels



    # for optimizer-setting
    if getattr(args, 'loss') is not None:
        opt['solver']['loss_type'] = args.loss

    # export CUDA_VISIBLE_DEVICES
    if torch.cuda.is_available():
        gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        print('===> Export CUDA_VISIBLE_DEVICES = [' + gpu_list + ']')
    else:
        print('===> CPU mode is set (NOTE: GPU is recommended)')

    # for datasets-setting
    scale = opt['scale']
    run_range = opt['run_range']
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        dataset['scale'] = scale
        dataset['run_range'] = run_range
        if args.datamode:
            dataset['mode'] = args.datamode

        if args.lrpath and 'train' in phase:
            if 'QB' in args.lrpath:
                dataset['name'] = 'QB'
            if 'WV2' in args.lrpath:
                dataset['name'] = 'WV2'
            if 'IK' in args.lrpath:
                dataset['name'] = 'IK'
            if 'WV3' in args.lrpath:
                dataset['name'] = 'WV3'
            if 'SP' in args.lrpath:
                dataset['name'] = 'SP'
            dataset['dataroot_LR'] = args.lrpath
            dataset['dataroot_HR'] = args.lrpath.replace('LR', 'HR')
            if args.panpath:
                dataset['dataroot_PAN'] = args.panpath
            else:
                dataset['dataroot_PAN'] = args.lrpath.replace('LRMS', 'LRPAN')
            if 'BIEDN' in network_opt['which_model']:
                dataset['dataroot_LRPAN'] = args.lrpath.replace('LRMS', "LLRPAN")
            if 'BDPN' in network_opt['which_model']:
                dataset['dataroot_LRPAN'] = args.lrpath.replace('LRMS', "MSX2")
        elif args.lrpath and 'val' in phase:
            dataset['dataroot_LR'] = args.lrpath.replace('train', 'value')
            dataset['dataroot_HR'] = args.lrpath.replace('LR', 'HR').replace('train', 'value')
            if args.panpath:
                dataset['dataroot_PAN'] = args.panpath.replace('train', 'value')
            else:
                dataset['dataroot_PAN'] = args.lrpath.replace('LRMS', 'LRPAN').replace('train', 'value')
            if 'BIEDN' in network_opt['which_model']:
                dataset['dataroot_LRPAN'] = args.lrpath.replace('LRMS', "LLRPAN").replace('train', 'value')
            if 'BDPN' in network_opt['which_model']:
                dataset['dataroot_LRPAN'] = args.lrpath.replace('LRMS', "MSX2").replace('train', 'value')
        elif args.lrpath and 'test' in phase:
            dataset['dataroot_LR'] = args.lrpath
            dataset['dataroot_HR'] = args.lrpath.replace('LR', 'HR')
            if args.panpath:
                dataset['dataroot_PAN'] = args.panpath
            else:
                dataset['dataroot_PAN'] = args.lrpath.replace('LRMS', 'LRPAN')
            if 'BIEDN' in network_opt['which_model']:
                dataset['dataroot_LRPAN'] = args.lrpath.replace('LRMS', "LLRPAN")
            if 'BDPN' in network_opt['which_model']:
                dataset['dataroot_LRPAN'] = args.lrpath.replace('LRMS', "MSX2")

    if getattr(args, 'save_name') is not None:
        config_str = args.save_name
    else:
        config_str = '%s_in%df%d_x%d' % (network_opt['which_model'].upper(), network_opt['in_channels'],
                                    network_opt['num_features'], opt['scale'])
    exp_path = os.path.join(os.getcwd(), 'experiments', config_str)

    if opt['is_train'] and opt['solver']['pretrain']:
        if 'pretrained_path' not in list(opt['solver'].keys()):
            raise ValueError("[Error] The 'pretrained_path' does not declarate in *.json")
        exp_path = os.path.dirname(os.path.dirname(opt['solver']['pretrained_path']))
        if opt['solver']['pretrain'] == 'finetune':
            exp_path += '_finetune'

    exp_path = os.path.relpath(exp_path)

    path_opt = OrderedDict()
    path_opt['exp_root'] = exp_path
    path_opt['epochs'] = os.path.join(exp_path, 'epochs')
    path_opt['visual'] = os.path.join(exp_path, 'visual')
    path_opt['records'] = os.path.join(exp_path, 'records')
    opt['path'] = path_opt

    if opt['is_train']:
        if opt['solver']['pretrain'] == 'resume':
            opt = dict_to_nonedict(opt)
        else:
            util.mkdir_and_rename(opt['path']['exp_root'])  # rename old experiments if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'exp_root'))
            save(opt, Dumper)
            opt = dict_to_nonedict(opt)

        print("===> Experimental DIR: [%s]" % exp_path)

    return opt


def save(opt, Dumper):
    dump_dir = opt['path']['exp_root']
    dump_path = os.path.join(dump_dir, 'options.yml')
    network_file = opt["networks"]['which_model'] + '.py'
    shutil.copy('./networks/' + network_file, os.path.join(dump_dir, network_file))
    with open(dump_path, 'w') as dump_file:
        yaml.dump(opt, dump_file, Dumper=Dumper)


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')