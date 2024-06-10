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
from torch.autograd import Variable
from pcdet.models.detectors.mae import MAE
from pcdet.utils.loss_utils import Lovasz_loss
import torch.nn.functional as F

class criterion(nn.Module):
    def __init__(self, config):
        super(criterion, self).__init__()
        self.config = config
        self.lambda_lovasz = self.config['train_params'].get('lambda_lovasz', 0.1)
        if 'seg_labelweights' in config['dataset_params']:
            seg_num_per_class = config['dataset_params']['seg_labelweights']
            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            seg_labelweights = torch.Tensor(np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0))
        else:
            seg_labelweights = None

        self.ce_loss = nn.CrossEntropyLoss(
            weight=seg_labelweights,
            ignore_index=config['dataset_params']['ignore_label']
        )
        self.lovasz_loss = Lovasz_loss(
            ignore=config['dataset_params']['ignore_label']
        )
    def forward(self, logits, labels):
        loss_main_ce = self.ce_loss(logits, labels.long())
        loss_main_lovasz = self.lovasz_loss(F.softmax(logits, dim=1), labels.long())
        loss_main = loss_main_ce + loss_main_lovasz * self.lambda_lovasz
        return loss_main_ce, loss_main_lovasz, loss_main
    
class MAE_Seg(nn.Module):
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

        num_classes = 20
        decoder_embed_dim = 512
        self.mae = MAE(model_cfg, num_class, dataset)

        self.img_size = (256, 1024)
        self.range_img_size = (64, 1024)

        self.img_mask_ratio, self.range_mask_ratio = 0, 0
        self.range_image_patch = (8, 128)

        decode_layer = [512, 256, 128, 64]
        self.decoder_pred = nn.ModuleList()
        for i in range(len(decode_layer) - 1):
            self.decoder_pred.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(decode_layer[i], decode_layer[i+1], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(decode_layer[i+1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(decode_layer[i+1], decode_layer[i+1], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(decode_layer[i+1]),
                    nn.ReLU(inplace=True)
                )
            )
        self.decoder_pred.append(
            nn.Sequential(
                nn.Conv2d(64, num_classes, kernel_size=1)
            )
        )
        print('model_cfg', model_cfg)
        self.criterion = criterion(model_cfg)
        self._init_mae_(model_cfg.PRETRAIN)
    
    def _init_mae_(self, pretrain=False):
        if pretrain:
            self.mae.load_params_from_file('/home/yanqiao/OpenPCDet/output/semantic_kitti_models/MAE/checkpoints/checkpoint_epoch_100.pth', None)
            print('Loaded MAE from pre-trained weights')
        else:
            print('vanilla training !')
    
    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    
    def forward_decode(self, x, ids_restore, x_mm, \
                        decoder_embed, \
                        mask_token, \
                        decoder_pos_embed):
        # embed tokens
        x = decoder_embed(x) # (B, L1, D=512)
        B, _, D = x.size()
        # append mask tokens to sequence
        mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) # (B, L2+1-L1=1024-256=768, D)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token # (B, L2, D)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle # (B, L2, D)
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token # (B, L2+1, D)
        # add pos embed
        x = x + decoder_pos_embed # (B, L2+1, D)
        x = x + x_mm
        x = x[:, 1:, :].reshape(B, self.range_image_patch[0], self.range_image_patch[1], D).permute(0, 3, 1, 2).contiguous()
        # apply Transformer blocks
        for blk in self.decoder_pred:
            x = blk(x)
        return x
    
    def forward_loss(self, pred, batch_dict):
        laser_range_in = batch_dict['laser_range_in'].permute(0,3,1,2).contiguous()
        laser_x = batch_dict['laser_x']
        laser_y = batch_dict['laser_y']
        pred_all = []
        for batch_idx in range(batch_dict['batch_size']):
            x_batch = laser_x[torch.where(laser_x[:, 0] == batch_idx)][:, -1]
            y_batch = laser_y[torch.where(laser_y[:, 0] == batch_idx)][:, -1]
            pre_batch = pred[batch_idx, :, y_batch.long(), x_batch.long()].permute(1, 0).contiguous()
            pred_all.append(pre_batch)
        pred_all = torch.cat(pred_all, dim=0)
        labels = batch_dict['point_labels']
        # from IPython import embed; embed()
        loss_main_ce, loss_main_lovasz, loss_main = self.criterion(pred_all, labels)
        return loss_main_ce, loss_main_lovasz, loss_main
  
    def forward(self, batch_dict):
        laser_range_in = batch_dict['laser_range_in'].permute(0,3,1,2).contiguous()
        laser_x = batch_dict['laser_x']
        laser_y = batch_dict['laser_y']
        raw_images = batch_dict['images']
        # B = batch_dict['batch_size']
        Batch_size, _, H_raw, W_raw = raw_images.size() # (256, 1024)

        images = torch.nn.functional.interpolate(raw_images, size=self.img_size, mode='bilinear')
        img_latent_full, img_mask_full, img_ids_restore_full = self.mae.image_encoder.forward_encoder(images, self.img_mask_ratio)
        range_latent_full, range_mask_full, range_ids_restore_full = self.mae.range_encoder.forward_encoder(laser_range_in, self.range_mask_ratio)

        img_latent_full_cls, img_mask_full_tokens_cls_unpatch, range_latent_full_cls, range_mask_full_tokens_cls_unpatch = self.mae.forward_patchfy_unpatchfy_img_range(img_latent_full, img_mask_full, img_ids_restore_full,
                                                                                                                        range_latent_full, range_mask_full, range_ids_restore_full)
        
        D = img_mask_full_tokens_cls_unpatch.shape[-1]
        h_img, w_img = self.mae.range_img_size[0] // self.mae.range_patch_size[0], self.mae.range_img_size[1] // self.mae.range_patch_size[1]
        img_token_2_range = Variable(torch.zeros((Batch_size, h_img, w_img, D), dtype=range_latent_full.dtype, device=range_latent_full.device))
        img_token_2_range[:, :, 48:80, :] = torch.nn.functional.interpolate(img_mask_full_tokens_cls_unpatch.permute(0, 3, 1, 2).contiguous(), size=(h_img, 32), mode='bilinear').permute(0, 2, 3, 1).contiguous().detach()
        img_token_2_range.requires_grad=True

        img_token_2_range = img_token_2_range.reshape(Batch_size, -1, D)
        img_token_2_range = torch.cat([img_latent_full_cls, img_token_2_range], 1)

        

        range_pred = self.forward_decode(range_latent_full, range_ids_restore_full, img_token_2_range,
                                        self.mae.range_encoder.decoder_embed,
                                        self.mae.range_encoder.mask_token,
                                        self.mae.range_encoder.decoder_pos_embed,
                                        )
        
        loss_main_ce, loss_main_lovasz, loss_main = self.forward_loss(range_pred, batch_dict)
        if self.training:
            tb_dict = {
                'loss_main_ce': loss_main_ce,
                'loss_main_lovasz': loss_main_lovasz,
            }
            ret_dict = {
                'loss': loss_main
            }
            return ret_dict, tb_dict, {}
        else:
            # inference at batch_size=1
            pred_dicts, recall_dicts = {}, {}
            return pred_dicts, recall_dicts
        
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
        

        


    