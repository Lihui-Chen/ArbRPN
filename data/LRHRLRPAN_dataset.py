import torch.utils.data as data

from data import common


class LRHRDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''

    def name(self):
        return self.opt['name']


    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.msx2 = True if 'MSX2' in opt['dataroot_LRPAN'] else False
        self.train = ('train' in opt['phase'])
        self.split = 'train' if self.train else 'test'
        self.scale = self.opt['scale']
        self.paths_HR, self.paths_LR = None, None

        # change the length of train dataset (influence the number of iterations in each epoch)
        self.repeat = 2

        # read image list from image/binary files
        self.paths_HR = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_HR'], opt['subset'])
        self.paths_LR = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_LR'], opt['subset'])
        self.paths_PAN = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_PAN'], opt['subset'])
        self.paths_LRPAN = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_LRPAN'], opt['subset'])

        assert self.paths_HR, '[Error] HR paths are empty.'
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                '[Error] HR: [%d] and LR: [%d] have different number of images.'%(
                len(self.paths_LR), len(self.paths_HR))

    def __getitem__(self, idx):
        lr, hr, lrpan, pan, lr_path = self._load_file(idx)
        if self.train:
            lr, hr, lrpan, pan= self._get_patch(lr, hr, lrpan, pan, msx2=self.msx2)
        lr_tensor, hr_tensor, lrpan_tensor, pan_tensor = common.np2Tensor([lr, hr, lrpan, pan], self.opt['run_range'], self.opt['img_range'])
        return {'LR': lr_tensor, 'HR': hr_tensor, 'LRPAN': lrpan_tensor, 'PAN':pan_tensor, 'LR_path': lr_path}

    def __len__(self):
        if self.train:
            return len(self.paths_LR)*self.repeat
        else:
            return len(self.paths_LR)


    def _get_index(self, idx):
        if self.train:
            return idx % len(self.paths_HR)
        else:
            return idx


    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr_path = self.paths_LR[idx]
        hr_path = self.paths_HR[idx]
        pan_path = self.paths_PAN[idx]
        lrpan_path = self.paths_LRPAN[idx]
        lr = common.read_img(lr_path, self.opt['data_type'])
        hr = common.read_img(hr_path, self.opt['data_type'])
        pan = common.read_img(pan_path, self.opt['data_type'])
        lrpan = common.read_img(lrpan_path, self.opt['data_type'])
        return lr, hr, lrpan, pan, lr_path


    def _get_patch(self, lr, hr, lrpan, pan, msx2=False):

        LR_size = self.opt['LR_size']
        # random crop and augment
        lr, hr, lrpan, pan = common.get_patch(lr, hr, pan,
                 LR_size, self.scale, lrpan=lrpan, msx2=msx2)
        lr, hr, lrpan, pan = common.augment([lr, hr, lrpan, pan])
        lr = common.add_noise(lr, self.opt['noise'])

        return lr, hr, lrpan, pan
