import os
from collections import OrderedDict
import pandas as pd
import scipy.misc as misc
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as thutil

from networks import create_model, init_weights
from .base_solver import BaseSolver
from utils import util

class SRSolver(BaseSolver):
    def __init__(self, opt):
        super(SRSolver, self).__init__(opt)
        self.train_opt = opt['solver']
        self.LR = self.Tensor()
        self.HR = self.Tensor()
        self.PAN = self.Tensor()
        self.LRPAN = self.Tensor()
        self.SR = None
        self.lmda = 1

        self.records = {'train_loss': [],
                        'val_loss': [],
                        'SAM':[],
                        'ERGAS':[],
                        'Q2n':[],
                        'CC':[],
                        'RMSE':[],
                        'lr': []}

        self.model = create_model(opt)
        self.print_network()

        if self.is_train:
            self.model.train()

            # set loss
            loss_type = self.train_opt['loss_type']
            if loss_type == 'l1':
                self.criterion_pix = nn.L1Loss()
            elif loss_type == 'l2':
                self.criterion_pix = nn.MSELoss()
            elif loss_type == 'myloss':
                my_net = importlib.import_module('networks.'+opt['networks']['which_model'])
                self.criterion_pix = my_net.myloss(opt['networks'])
            else:
                raise NotImplementedError('Loss type [%s] is not implemented!'%loss_type)

            if self.use_gpu:
                self.criterion_pix = self.criterion_pix.cuda()

            # set optimizer
            weight_decay = self.train_opt['weight_decay'] if self.train_opt['weight_decay'] else 0
            optim_type = self.train_opt['type'].upper()
            if optim_type == "ADAM":
                self.optimizer = optim.Adam(self.model.parameters(),
                                            lr=self.train_opt['learning_rate'], weight_decay=weight_decay)
            else:
                raise NotImplementedError('Loss type [%s] is not implemented!' % optim_type)

            # set lr_scheduler
            if self.train_opt['lr_scheme'].lower() == 'multisteplr':
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                self.train_opt['lr_steps'],
                                                                self.train_opt['lr_gamma'])
            else:
                raise NotImplementedError('Only MultiStepLR scheme is supported!')

        self.load()

        print('===> Solver Initialized : [%s] || Use CL : [%s] || Use GPU : [%s]'%(self.__class__.__name__,
                                                                        self.use_cl, self.use_gpu))
        if self.is_train:
            print("optimizer: ", self.optimizer)
            print("lr_scheduler milestones: %s   gamma: %f"%(self.scheduler.milestones, self.scheduler.gamma))

    def _net_init(self, init_type='kaiming'):
        print('==> Initializing the network using [%s]'%init_type)
        # init_weights(self.model, init_type, scale=0.1)


    def feed_data(self, batch, need_HR=True):
        input = batch['LR']
        input_ = batch['PAN']
        self.LR.resize_(input.size()).copy_(input)
        self.PAN.resize_(input_.size()).copy_(input_)
        if 'BIEDN' == self.opt['networks']['which_model']:# or 'BDPN' == self.opt['networks']['which_model']:
            LRPAN = batch['LRPAN']
            self.LRPAN.resize_(LRPAN.shape).copy_(LRPAN)

        if need_HR:
            target = batch['HR']
            self.HR.resize_(target.size()).copy_(target)

    def train_step(self, mask=None):
        self.model.train()
        self.optimizer.zero_grad()
        if 'mask' in self.opt['networks']['which_model']:
            output = self.model(self.LR, self.PAN, mask=mask)
            loss = self.criterion_pix(output, self.HR, mask)
        elif 'BIEDN' in self.opt['networks']['which_model']:
            output = self.model(self.LR, self.LRPAN, self.PAN)
            loss = self.criterion_pix(output, self.HR)
        else:
            output = self.model(self.LR, self.PAN)
            loss = self.criterion_pix(output, self.HR)
        loss.backward()
        loss = loss.item()

        # for stable training
        if loss < self.skip_threshold * self.last_epoch_loss:
            self.optimizer.step()
            self.last_epoch_loss = loss
        else:
            print('[Warning] Skip this batch! (Loss: {})'.format(loss))

        self.model.eval()
        return loss

    def test(self):
        self.model.eval()
        with torch.no_grad():
            forward_func = self._overlap_crop_forward if self.use_chop else self.model.forward
            if 'BIEDN' in self.opt['networks']['which_model']:
                SR = self.model(self.LR, self.LRPAN, self.PAN,)
            else:
                SR = forward_func(self.LR, self.PAN)

            if isinstance(SR, list):
                self.SR = SR[-1]
            else:
                self.SR = SR

        self.model.train()
        if self.is_train:
            if 'BDPN' in self.opt['networks']['which_model']:
                    loss_pix = self.criterion_pix(self.SR, self.LRPAN, self.HR)
            else:
                loss_pix = self.criterion_pix(self.SR, self.HR)
            return loss_pix.item()

    def _overlap_crop_forward(self, x, pan, shave=10, min_size=100000, bic=None):
        """
        chop for less memory consumption during test
        """
        n_GPUs = 2
        scale = self.scale
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]
        
        pan_h_size = h_size*scale
        pan_w_size = w_size*scale
        pan_h = h*scale
        pan_w = w*scale
        
        pan_list = [
            pan[:, :, 0:pan_h_size, 0:pan_w_size],
            pan[:, :, 0:pan_h_size, (pan_w - pan_w_size):pan_w],
            pan[:, :, (pan_h - pan_h_size):pan_h, 0:pan_w_size],
            pan[:, :, (pan_h - pan_h_size):pan_h, (pan_w - pan_w_size):pan_w]]

        if bic is not None:
            bic_h_size = h_size*scale
            bic_w_size = w_size*scale
            bic_h = h*scale
            bic_w = w*scale
            bic_list = [
                bic[:, :, 0:bic_h_size, 0:bic_w_size],
                bic[:, :, 0:bic_h_size, (bic_w - bic_w_size):bic_w],
                bic[:, :, (bic_h - bic_h_size):bic_h, 0:bic_w_size],
                bic[:, :, (bic_h - bic_h_size):bic_h, (bic_w - bic_w_size):bic_w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                pan_batch = torch.cat(pan_list[i:(i + n_GPUs)], dim=0)
                if bic is not None:
                    bic_batch = torch.cat(bic_list[i:(i + n_GPUs)], dim=0)

                sr_batch_temp = self.model(lr_batch, pan_batch, mask)

                if isinstance(sr_batch_temp, (list, tuple)):
                    sr_batch = sr_batch_temp[-1]
                else:
                    sr_batch = sr_batch_temp

                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self._overlap_crop_forward(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
                ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def save_checkpoint(self, epoch, is_best):
        """
        save checkpoint to experimental dir
        """
        filename = os.path.join(self.checkpoint_dir, 'last_ckp.pth')
        print('===> Saving last checkpoint to [%s] ...]'%filename)
        ckp = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
            'best_epoch': self.best_epoch,
            'records': self.records
        }
        torch.save(ckp, filename)
        if is_best:
            print('===> Saving best checkpoint to [%s] ...]' % filename.replace('last_ckp','best_ckp'))
            torch.save(ckp, filename.replace('last_ckp','best_ckp'))

        if epoch % self.train_opt['save_ckp_step'] == 0:
            print('===> Saving checkpoint [%d] to [%s] ...]' % (epoch,
                                                                filename.replace('last_ckp','epoch_%d_ckp.pth'%epoch)))

            torch.save(ckp, filename.replace('last_ckp','epoch_%d_ckp.pth'%epoch))

    def load(self):
        """
        load or initialize network
        """
        if (self.is_train and self.opt['solver']['pretrain']) or not self.is_train:
            model_path = self.opt['solver']['pretrained_path']
            if model_path is None: raise ValueError("[Error] The 'pretrained_path' does not declarate in *.json")

            print('===> Loading model from [%s]...' % model_path)
            if self.is_train:
                checkpoint = torch.load(model_path)
                self.model.load_state_dict(checkpoint['state_dict'])

                if self.opt['solver']['pretrain'] == 'resume':
                    self.cur_epoch = checkpoint['epoch'] + 1
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.best_pred = checkpoint['best_pred']
                    self.best_epoch = checkpoint['best_epoch']
                    self.records = checkpoint['records']

            else:
                checkpoint = torch.load(model_path)
                if 'state_dict' in checkpoint.keys(): checkpoint = checkpoint['state_dict']
                load_func = self.model.load_state_dict
                load_func(checkpoint)

        else:
            self._net_init()

    def get_current_visual(self, need_np=True, need_HR=True):
        """
        return LR SR (HR) images
        """
        if isinstance(self.SR, (tuple, list)):
            self.SR = self.SR[-1]
        out_dict = OrderedDict()
        # out_dict['LR'] = self.LR.data[0].float().cpu()
        out_dict['SR'] = self.SR.data[0].float().cpu()
        if need_np:  out_dict['SR'] = util.pan_Tensor2np([out_dict['SR']],
                                                                        self.opt['run_range'], self.opt['img_range'])[0]
        if need_HR:
            out_dict['HR'] = self.HR.data[0].float().cpu()
            if need_np: out_dict['HR'] = util.pan_Tensor2np([out_dict['HR']],
                                                            self.opt['run_range'], self.opt['img_range'])[0]
        return out_dict

    def save_current_visual(self, epoch, iter):
        """
        save visual results for comparison
        """
        if epoch % self.save_vis_step == 0:
            visuals_list = []
            visuals = self.get_current_visual(need_np=False)
            visuals_list.extend([util.quantize(visuals['HR'].squeeze(0), self.opt['run_range']),
                                util.quantize(visuals['SR'].squeeze(0), self.opt['run_range'])])
            visual_images = torch.stack(visuals_list)
            visual_images = thutil.make_grid(visual_images, nrow=2, padding=5)
            visual_images = visual_images.byte().permute(1, 2, 0).numpy()
            misc.imsave(os.path.join(self.visual_dir, 'epoch_%d_img_%d.png' % (epoch, iter + 1)),
                        visual_images)

    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def update_learning_rate(self, epoch):
        self.scheduler.step()

    def get_current_log(self):
        log = OrderedDict()
        log['epoch'] = self.cur_epoch
        log['best_pred'] = self.best_pred
        log['best_epoch'] = self.best_epoch
        log['records'] = self.records
        return log

    def set_current_log(self, log):
        self.cur_epoch = log['epoch']
        self.best_pred = log['best_pred']
        self.best_epoch = log['best_epoch']
        self.records = log['records']

    def save_current_log(self):
        for key in self.records.keys():
            self.records[key].append(self.records[key][self.best_epoch - 1])
        res_index = list(range(1, self.cur_epoch + 1))
        res_index.append('Best epoch' + str(self.best_epoch))
        data_frame = pd.DataFrame(
            data={key:value for key, value in self.records.items()},
            index=res_index
        )
        data_frame.to_csv(os.path.join(self.records_dir, 'train_records.csv'),
                        index_label='epoch')
        for key in self.records.keys():
            self.records[key].pop()

    def print_network(self):
        """
        print network summary including module and number of parameters
        """
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.model.__class__.__name__,
                                                 self.model.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.model.__class__.__name__)

        print("==================================================")
        print("===> Network Summary\n")
        net_lines = []
        line = s + '\n'
        print(line)
        net_lines.append(line)
        line = 'Network structure: [{}], with parameters: [{:,d}]'.format(net_struc_str, n)
        print(line)
        net_lines.append(line)

        if self.is_train:
            with open(os.path.join(self.exp_root, 'network_summary.txt'), 'w') as f:
                f.writelines(net_lines)

        print("==================================================")
