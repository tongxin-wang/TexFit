import logging
import math
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import clip
import os

from models.archs.fcn_arch import FCNHead
from models.archs.unet_arch import AttrUNet
from models.losses.accuracy import accuracy
from models.losses.cross_entropy_loss import CrossEntropyLoss

logger = logging.getLogger('base')


class ERLM():
    """Editing Region Generation model.
    """

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda')
        self.is_train = opt['is_train']

        clip_model, _ = clip.load('ViT-B/32', device=torch.device("cpu"))
        self.clip = clip_model.to(self.device)
        self.encoder = AttrUNet(
            in_channels=opt['encoder_in_channels'], attr_embedding=opt['text_embedding_dim']).to(self.device)
        self.decoder = FCNHead(
            in_channels=opt['fc_in_channels'],
            in_index=opt['fc_in_index'],
            channels=opt['fc_channels'],
            num_convs=opt['fc_num_convs'],
            concat_input=opt['fc_concat_input'],
            dropout_ratio=opt['fc_dropout_ratio'],
            num_classes=opt['fc_num_classes'],
            align_corners=opt['fc_align_corners'],
        ).to(self.device)

        self.init_training_settings()
        self.palette = [[0, 0, 0], [255, 255, 255]]

    def init_training_settings(self):
        optim_params = []

        for v in self.encoder.parameters():
            if v.requires_grad:
                optim_params.append(v)
        for v in self.decoder.parameters():
            if v.requires_grad:
                optim_params.append(v)
        # set up optimizers
        self.optimizer = torch.optim.Adam(
            optim_params,
            self.opt['lr'],
            weight_decay=self.opt['weight_decay'])
        self.log_dict = OrderedDict()
        self.entropy_loss = CrossEntropyLoss().to(self.device)

    def feed_data(self, data):
        self.image = data['image'].to(self.device)
        self.mask = data['mask'].to(self.device)
        text = data['text']
        text_inputs = torch.cat([clip.tokenize(text)]).to(self.device)
        with torch.no_grad():
            self.text = self.clip.encode_text(text_inputs)

    def optimize_parameters(self):
        self.encoder.train()
        self.decoder.train()

        self.text_enc = self.encoder(self.image, self.text)
        self.seg_logits = self.decoder(self.text_enc)

        loss = self.entropy_loss(self.seg_logits, self.mask)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_dict['loss_total'] = loss

    def inference(self, data_loader, save_dir):
        self.encoder.eval()
        self.decoder.eval()

        acc = 0
        num = 0

        for _, data in enumerate(data_loader):
            image = data['image'].to(self.device)
            text = data['text']
            text_inputs = torch.cat([clip.tokenize(text)]).to(self.device)
            mask = data['mask'].to(self.device)
            img_name = data['img_name']

            num += image.size(0)
            with torch.no_grad():
                text_embedding = self.clip.encode_text(text_inputs)
                text_enc = self.encoder(image, text_embedding)
                seg_logits = self.decoder(text_enc)
            seg_pred = seg_logits.argmax(dim=1)
            acc += accuracy(seg_logits, mask)
            palette_label = self.palette_result(mask.cpu().numpy())
            palette_pred = self.palette_result(seg_pred.cpu().numpy())
            image_numpy = image[0].cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
            image_numpy = image_numpy[..., ::-1]
            concat_result = np.concatenate(
                (image_numpy, palette_pred, palette_label), axis=1)
            img_name_base, img_name_ext = os.path.splitext(img_name[0])
            mmcv.imwrite(concat_result, f'{save_dir}/{img_name_base}_{text[0]}{img_name_ext}')

        self.encoder.train()
        self.decoder.train()
        return (acc / num).item()

    def get_current_log(self):
        return self.log_dict

    def update_learning_rate(self, epoch):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int): Warmup iter numbers. -1 for no warmup.
                Default: -1.
        """
        lr = self.optimizer.param_groups[0]['lr']

        if self.opt['lr_decay'] == 'step':
            lr = self.opt['lr'] * (
                self.opt['gamma']**(epoch // self.opt['step']))
        elif self.opt['lr_decay'] == 'cos':
            lr = self.opt['lr'] * (
                1 + math.cos(math.pi * epoch / self.opt['num_epochs'])) / 2
        elif self.opt['lr_decay'] == 'linear':
            lr = self.opt['lr'] * (1 - epoch / self.opt['num_epochs'])
        elif self.opt['lr_decay'] == 'linear2exp':
            if epoch < self.opt['turning_point'] + 1:
                # learning rate decay as 95%
                # at the turning point (1 / 95% = 1.0526)
                lr = self.opt['lr'] * (
                    1 - epoch / int(self.opt['turning_point'] * 1.0526))
            else:
                lr *= self.opt['gamma']
        elif self.opt['lr_decay'] == 'schedule':
            if epoch in self.opt['schedule']:
                lr *= self.opt['gamma']
        else:
            raise ValueError('Unknown lr mode {}'.format(self.opt['lr_decay']))
        # set learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def save_network(self, save_path):
        """Save networks.
        """

        save_dict = {}
        save_dict['encoder'] = self.encoder.state_dict()
        save_dict['decoder'] = self.decoder.state_dict()

        torch.save(save_dict, save_path)

    def load_network(self):
        checkpoint = torch.load(self.opt['pretrained_model_path'])

        self.encoder.load_state_dict(
            checkpoint['encoder'], strict=True)
        self.encoder.eval()

        self.decoder.load_state_dict(
            checkpoint['decoder'], strict=True)
        self.decoder.eval()

    def palette_result(self, result):
        seg = result[0]
        palette = np.array(self.palette)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]
        return color_seg
