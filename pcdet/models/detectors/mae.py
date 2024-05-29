import os

import torch
import torch.nn as nn
import numpy as np
from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils.spconv_utils import find_all_spconv_keys
from .. import backbones_2d, backbones_3d, dense_heads, roi_heads
from ..backbones_2d import map_to_bev
from ..backbones_3d import pfe, vfe
from ..model_utils import model_nms_utils
from .model_mae import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14


class MAE(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]
        self.range_mask_ratio, self.img_mask_ratio = 0.75, 0.75
        self.img_size = (64, 1024)

        self.range_encoder = mae_vit_large_patch16(in_chans=5, img_with_size=self.img_size, out_chans=5)
        self.image_encoder = mae_vit_large_patch16(in_chans=3, img_with_size=self.img_size, out_chans=3)

        self.img_gen_decoder = nn.ModuleList([
            nn.ConvTranspose2d(3, 3, kernel_size=(9, 1), stride=(4, 1), padding=(3, 0), output_padding=(1, 0), bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        ])
        # self.range_gen_decoder = nn.ModuleList([
        #     nn.ConvTranspose2d(3, 3, kernel_size=(9, 1), stride=(4, 1), padding=(3, 0), output_padding=(1, 0), bias=False),
        #     nn.BatchNorm2d(3),
        #     nn.ReLU(True)
        # ])

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    
    def forward(self, batch_dict):
        # raw image & range image shape: (B, 3, 376, 1241)
        laser_range_in = batch_dict['laser_range_in'].permute(0,3,1,2).contiguous()
        laser_x = batch_dict['laser_x']
        laser_y = batch_dict['laser_y']
        images = batch_dict['images']
        B = batch_dict['batch_size']
        _, _, H_raw, W_raw = images.size() # (256, 1024)

        images = torch.nn.functional.interpolate(images, size=self.img_size, mode='bilinear')

        laser_range_in_roi = laser_range_in[:, :, :, 384:640] # (64, 256)
        laser_range_in_roi = torch.nn.functional.interpolate(laser_range_in_roi, size=self.img_size, mode='bilinear') # (64, 1024)
        
        # img1 = images[0,:,:,:].permute(1,2,0).detach().cpu().numpy()
        # range1 = laser_range_in_roi[0,:,:,:].permute(1,2,0).detach().cpu().numpy()
        # import cv2
        # img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX)
        # range1 = cv2.normalize(range1[:,:,0], None, 0, 255, cv2.NORM_MINMAX)
        # cv2.imwrite('./img1.png', img1)
        # cv2.imwrite('./range1.png', range1)

        # color image encoding
        img_latent, img_mask, img_ids_restore = self.image_encoder.forward_encoder(images, self.img_mask_ratio)
        img_latent_full, _, _ = self.image_encoder.forward_encoder(images, 0)
        # img_latent: (B, L=H*W / 16 / 16 * (1-mask_ratio) + 1=256+1=257, C=1024)
        # img_mask: (B, L=H*W/16/16=1024)
        # img_ids_restore: (B, L=H*W/16/16=1024)

        # range ROI image encoding
        range_latent_roi, range_mask_roi, range_ids_restore_roi = self.range_encoder.forward_encoder(laser_range_in_roi, self.range_mask_ratio)
        range_latent_roi_full, _, _ = self.range_encoder.forward_encoder(laser_range_in_roi, 0)
        

        
        # range ROI image encoding
        img_pred = self.image_encoder.forward_decoder(img_latent, img_ids_restore)
        # range_latent: (B, L=H*W / 16 / 16 * (1-mask_ratio) + 1=64+1=65, C=1024)
        # range_mask: (B, L=H*W/16/16=256)
        # range_ids_restore: (B, L=H*W/16/16=256)

        # img + full range ROI
        img_pred = self.forward_decoder(img_latent, img_ids_restore, self.range_encoder.decoder_embed(range_latent_roi_full),
                                        self.image_encoder.decoder_embed,
                                        self.image_encoder.mask_token,
                                        self.image_encoder.decoder_pos_embed,
                                        self.image_encoder.decoder_blocks,
                                        self.image_encoder.decoder_norm,
                                        self.image_encoder.decoder_pred)
        img_loss = self.image_encoder.forward_loss(images, img_pred, img_mask)


        
        # range ROI + full image
        range_roi_pred = self.forward_decoder(range_latent_roi, range_ids_restore_roi, self.image_encoder.decoder_embed(img_latent_full),
                                        self.range_encoder.decoder_embed,
                                        self.range_encoder.mask_token,
                                        self.range_encoder.decoder_pos_embed,
                                        self.range_encoder.decoder_blocks,
                                        self.range_encoder.decoder_norm,
                                        self.range_encoder.decoder_pred)
        

        # from IPython import embed; embed()
        
        range_roi_loss = self.range_encoder.forward_loss(laser_range_in_roi, range_roi_pred, range_mask_roi)

        range_latent, range_mask, range_ids_restore = self.range_encoder.forward_encoder(laser_range_in, self.range_mask_ratio)
        range_pred = self.range_encoder.forward_decoder(range_latent, range_ids_restore)

        range_loss = self.range_encoder.forward_loss(laser_range_in, range_pred, range_mask)

        loss = range_roi_loss + range_loss + img_loss

        # from IPython import embed; embed()
        if self.training:
            tb_dict = {
                'range_roi_loss': range_roi_loss,
                'range_loss': range_loss,
                'img_loss': img_loss,
            }
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, {}
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts
        
        return batch_dict
    
    def forward_decoder(self, x, ids_restore, x_mm, \
                        decoder_embed, \
                        mask_token, \
                        decoder_pos_embed,\
                        decoder_blocks, \
                        decoder_norm,\
                        decoder_pred):# x, ids_restore: (B, L1=H*W/16/16*(1-mask_ratio)+1=257, C) , (B, L2=H*W/16/16=1024)
        # embed tokens
        x = decoder_embed(x) # (B, L1, D=512)

        # append mask tokens to sequence
        mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) # (B, L2+1-L1=1024-256=768, D)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token # (B, L2, D)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle # (B, L2, D)
        
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token # (B, L2+1, D)

        # add pos embed
        x = x + decoder_pos_embed # (B, L2+1, D)

        x = x + x_mm
        
        # apply Transformer blocks
        for blk in decoder_blocks:
            x = blk(x)
        # x: (B, L2+1, D)
        x = decoder_norm(x)

        # predictor projection
        x = decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


    def get_training_loss(self):
        loss = 0
        disp_dict = {}
        tb_dict = {}
        return loss, tb_dict, disp_dict
    
    def post_processing(self, batch_dict):
        pred_dicts, recall_dict= {}, {}
        return pred_dicts, recall_dict


    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def load_params_from_file(self, filename, logger, to_cpu=False, pre_trained_path=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        if not pre_trained_path is None:
            pretrain_checkpoint = torch.load(pre_trained_path, map_location=loc_type)
            pretrain_model_state_disk = pretrain_checkpoint['model_state']
            model_state_disk.update(pretrain_model_state_disk)
            
        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch
