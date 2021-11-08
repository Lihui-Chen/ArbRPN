import torch.utils.data as data

from data import common


class LRDataset(data.Dataset):
    '''
    Read LR images only in test phase.
    '''

    def name(self):
        return common.find_benchmark(self.opt['dataroot_LR'])


    def __init__(self, opt):
        super(LRDataset, self).__init__()
        self.opt = opt
        self.scale = self.opt['scale']
        self.paths_LR = None
        self.paths_PAN = None
        self.paths_LRPAN = None

        # read image list from image/binary files
        self.paths_LR = common.get_image_paths(opt['data_type'], opt['dataroot_LR'])
        self.paths_PAN = common.get_image_paths(opt['data_type'], opt['dataroot_PAN'])
        self.paths_LRPAN = common.get_image_paths(opt['data_type'], opt['dataroot_LRPAN'])
        assert self.paths_LR, '[Error] LR paths are empty.'
        assert (len(self.paths_LR)==len(self.paths_PAN)), 'LRMS is not equal to PAN.'


    def __getitem__(self, idx):
        # get LR image
        lr, pan, lrpan, lr_path = self._load_file(idx)
        lr_tensor, pan_tensor, lrpan_tensor = common.np2Tensor([lr, pan], self.opt['rgb_range'])[0]
        return {'LR': lr_tensor, 'PAN':pan_tensor, 'LRPAN':lrpan_tensor, 'LR_path': lr_path}


    def __len__(self):
        return len(self.paths_LR)


    def _load_file(self, idx):
        lr_path = self.paths_LR[idx]
        pan_path = self.paths_PAN[idx]
        lrpan_path = self.paths_LRPAN[idx]
        lr = common.read_img(lr_path, self.opt['data_type'])
        pan = common.read_img(pan_path, self.opt['data_type'])
        lrpan = common.read_img(lrpan_path, self.opt['data_type'])

        return lr, pan, lrpan, lr_path