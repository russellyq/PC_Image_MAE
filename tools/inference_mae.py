import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from skimage import io
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
from pcdet.datasets.kitti.laserscan import LaserScan
import matplotlib.pyplot as plt
import cv2


laserscaner = LaserScan()

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=5, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False, help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    parser.add_argument('--use_amp', action='store_true', help='use mix precision training')
    

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    
    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg
def select_points_in_frustum(points_2d, x1, y1, x2, y2):
        """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param points_3d: point cloud
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
        keep_ind = (points_2d[:, 0] > x1) * \
                   (points_2d[:, 1] > y1) * \
                   (points_2d[:, 0] < x2) * \
                   (points_2d[:, 1] < y2)

        return keep_ind

def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)

        return calib_out #, calibration_kitti.Calibration(calib_path)

def collect_data(index, dataset_cfg, device):
    batch_dict = {}
    root_dir = '/media/yanqiao/Seagate Hub/semantic-kitti/dataset/sequences/08/'
    velo_dir = root_dir + 'velodyne/' + '%06d.bin' % index
    img_dir = root_dir + 'image_2/' + '%06d.png' % index
    calib_dir = root_dir + 'calib.txt'
    calib = read_calib(calib_dir)
    proj_matrix = np.matmul(calib["P2"], calib["Tr"])

    raw_data = np.fromfile(velo_dir, dtype=np.float32).reshape((-1, 4))
    points = raw_data[:, :3]
    image = io.imread(img_dir)
    image = image.astype(np.float32)
    image /= 255.0

    data = raw_data.copy()
    xyz = points.copy()
    ref_pc = xyz.copy()
    ref_index = np.arange(len(ref_pc))

    mask_x = np.logical_and(xyz[:, 0] > dataset_cfg['dataset_params']['min_volume_space'][0], xyz[:, 0] < dataset_cfg['dataset_params']['max_volume_space'][0])
    mask_y = np.logical_and(xyz[:, 1] > dataset_cfg['dataset_params']['min_volume_space'][1], xyz[:, 1] < dataset_cfg['dataset_params']['max_volume_space'][1])
    mask_z = np.logical_and(xyz[:, 2] > dataset_cfg['dataset_params']['min_volume_space'][2], xyz[:, 2] < dataset_cfg['dataset_params']['max_volume_space'][2])
    mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

    data = data[mask]
    xyz = xyz[mask]
    ref_pc = ref_pc[mask]
    ref_index = ref_index[mask]

    # project points into image
    keep_idx = xyz[:, 0] > 0  # only keep point in front of the vehicle
    points_hcoords = np.concatenate([xyz[keep_idx], np.ones([keep_idx.sum(), 1], dtype=np.float32)], axis=1)
    img_points = (proj_matrix @ points_hcoords.T).T
    img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points INTO shape: (N, 2)
    keep_idx_img_pts = select_points_in_frustum(img_points, 0, 0, image.shape[0], image.shape[1])
    keep_idx[keep_idx] = keep_idx_img_pts

    # fliplr so that indexing is row, col and not col, row
    img_points = np.fliplr(img_points)
    points_img = img_points[keep_idx_img_pts]
    
    point2img_index = np.arange(len(xyz))[keep_idx]

    # crop image for processing:
    left = ( image.shape[1] - dataset_cfg['dataset_params']['bottom_crop'][0] ) // 2
    right = left + dataset_cfg['dataset_params']['bottom_crop'][0] 
    top = image.shape[0] - dataset_cfg['dataset_params']['bottom_crop'][1]
    bottom = image.shape[0]

    # update image points
    keep_idx = points_img[:, 0] >= top
    keep_idx = np.logical_and(keep_idx, points_img[:, 0] < bottom)
    keep_idx = np.logical_and(keep_idx, points_img[:, 1] >= left)
    keep_idx = np.logical_and(keep_idx, points_img[:, 1] < right)

    # crop image
    image = image[top:bottom, left:right, :]

    points_img = points_img[keep_idx]
    points_img[:, 0] -= top
    points_img[:, 1] -= left

    point2img_index = point2img_index[keep_idx]

    out_dict = laserscaner.open_scan(data)
    laser_range = out_dict['range']
    laser_ori_xyz = out_dict['ori_xyz']
    laser_ori_r = out_dict['ori_r']
    laser_idx = out_dict['idx']
    laser_mask = out_dict['mask']
    laser_range_in = out_dict['range_in']
    laser_y = out_dict['y']
    laser_x = out_dict['x']
    laser_points = out_dict['points']

    batch_dict['laser_range_in'] = torch.from_numpy(laser_range_in).unsqueeze(dim=0).to(device)

    laser_x = np.concatenate([0 * np.ones((laser_x.shape[0], 1)),laser_x.reshape(-1, 1)], -1).astype(np.float32)
    laser_y = np.concatenate([0 * np.ones((laser_y.shape[0], 1)),laser_y.reshape(-1, 1)], -1).astype(np.float32)
    batch_dict['laser_x'] = torch.from_numpy(laser_x).unsqueeze(dim=0).to(device)
    batch_dict['laser_y'] = torch.from_numpy(laser_y).unsqueeze(dim=0).to(device)

    batch_dict['images'] = torch.from_numpy(image).unsqueeze(dim=0).permute(0, 3, 1, 2).contiguous().to(device)
    
    print('img: ', batch_dict['images'].size())
    print('laser_range_in: ', batch_dict['laser_range_in'].size())
    return batch_dict

def load_params_with_optimizer(model, filename, to_cpu=False):
    if not os.path.isfile(filename):
            raise FileNotFoundError
    loc_type = torch.device('cpu') if to_cpu else None
    checkpoint = torch.load(filename, map_location=loc_type)
    model._load_state_dict(checkpoint['model_state'], strict=True)
    return model

def main():
    args, cfg = parse_config()

    device = torch.device('cuda:0')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=None)
    model = load_params_with_optimizer(model, '/home/yanqiao/OpenPCDet/output/semantic_kitti_models/ckpt/checkpoint_epoch_100.pth')
    model.to(device)
    model.training = False
    root_dir = '/media/yanqiao/Seagate Hub/semantic-kitti/dataset/sequences/08/pred/'
    os.makedirs(root_dir, exist_ok=True)
    with torch.no_grad():
        for i in range(10):
            batch_dict = collect_data(i, cfg.DATA_CONFIG, device)
            pred_dicts, _ = model(batch_dict)

            # saving img parts
            plt.rcParams['figure.figsize'] = [32, 32]
            plt.clf()

            plt.subplot(5, 1, 1)
            img_raw = pred_dicts['img']
            img_raw = cv2.normalize(img_raw, None, 0, 1, cv2.NORM_MINMAX)
            plt.imshow(img_raw)
            plt.title('raw_img', fontsize=16)
            plt.axis('off')

            plt.subplot(5, 1, 2)
            img_raw = pred_dicts['img_raw']
            img_raw = cv2.normalize(img_raw, None, 0, 1, cv2.NORM_MINMAX)
            plt.imshow(img_raw)
            plt.title('raw_interpolate', fontsize=16)
            plt.axis('off')

            plt.subplot(5, 1, 3)
            im_masked = pred_dicts['im_masked']
            im_masked = cv2.normalize(im_masked, None, 0, 1, cv2.NORM_MINMAX)
            plt.imshow(im_masked)
            plt.title('masked', fontsize=16)
            plt.axis('off')

            plt.subplot(5, 1, 4)
            img_pred = pred_dicts['img_pred']
            img_pred = cv2.normalize(img_pred, None, 0, 1, cv2.NORM_MINMAX)
            plt.imshow(img_pred)
            plt.title('reconstruction', fontsize=16)
            plt.axis('off')

            plt.subplot(5, 1, 5)
            im_paste = pred_dicts['im_paste']
            im_paste = cv2.normalize(im_paste, None, 0, 1, cv2.NORM_MINMAX)
            plt.imshow(im_paste)
            plt.title('reconstruction + visible', fontsize=16)
            plt.axis('off')

            file_name  = root_dir + 'IMG_' + '%06d.png' % i
            plt.savefig(file_name)


            # saving range parts
            plt.rcParams['figure.figsize'] = [48, 48]
            plt.clf()

            plt.subplot(6, 1, 1)
            img_raw = pred_dicts['img']
            img_raw = cv2.normalize(img_raw, None, 0, 1, cv2.NORM_MINMAX)
            plt.imshow(img_raw)
            plt.title('raw_interpolate', fontsize=16)
            plt.axis('off')
            
            plt.subplot(6, 1, 2)
            img_raw = pred_dicts['img_raw']
            img_raw = cv2.normalize(img_raw, None, 0, 1, cv2.NORM_MINMAX)
            plt.imshow(img_raw)
            plt.title('raw_interpolate', fontsize=16)
            plt.axis('off')

            plt.subplot(6, 1, 3)
            range_roi_raw = pred_dicts['range_roi_raw'][:, :, 0]
            range_roi_raw = cv2.normalize(range_roi_raw, None, 0, 1, cv2.NORM_MINMAX)
            plt.imshow(range_roi_raw)
            plt.title('range_roi_raw', fontsize=16)
            plt.axis('off')

            plt.subplot(6, 1, 4)
            range_roi_masked = pred_dicts['range_roi_masked'][:, :, 0]
            range_roi_masked = cv2.normalize(range_roi_masked, None, 0, 1, cv2.NORM_MINMAX)
            plt.imshow(range_roi_masked)
            plt.title('range_roi_masked', fontsize=16)
            plt.axis('off')

            plt.subplot(6, 1, 5)
            range_roi_pred = pred_dicts['range_roi_pred'][:, :, 0]
            range_roi_pred = cv2.normalize(range_roi_pred, None, 0, 1, cv2.NORM_MINMAX)
            plt.imshow(range_roi_pred)
            plt.title('reconstruction', fontsize=16)
            plt.axis('off')

            plt.subplot(6, 1, 6)
            range_roi_paste = pred_dicts['range_roi_paste'][:, :, 0]
            range_roi_paste = cv2.normalize(range_roi_paste, None, 0, 1, cv2.NORM_MINMAX)
            plt.imshow(range_roi_paste)
            plt.title('reconstruction + visible', fontsize=16)
            plt.axis('off')

            file_name  = root_dir + 'RANGE_ROI_' + '%06d.png' % i
            plt.savefig(file_name)


            # saving range full parts
            plt.rcParams['figure.figsize'] = [48, 48]
            plt.clf()

            plt.subplot(6, 1, 1)
            img_raw = pred_dicts['img']
            img_raw = cv2.normalize(img_raw, None, 0, 1, cv2.NORM_MINMAX)
            plt.imshow(img_raw)
            plt.title('raw_interpolate', fontsize=16)
            plt.axis('off')
            
            plt.subplot(6, 1, 2)
            img_raw = pred_dicts['img_raw']
            img_raw = cv2.normalize(img_raw, None, 0, 1, cv2.NORM_MINMAX)
            plt.imshow(img_raw)
            plt.title('raw_interpolate', fontsize=16)
            plt.axis('off')

            plt.subplot(6, 1, 3)
            range_raw = pred_dicts['range_raw'][:, :, 0]
            range_raw = cv2.normalize(range_raw, None, 0, 1, cv2.NORM_MINMAX)
            plt.imshow(range_raw)
            plt.title('range_raw', fontsize=16)
            plt.axis('off')

            plt.subplot(6, 1, 4)
            range_masked = pred_dicts['range_masked'][:, :, 0]
            range_masked = cv2.normalize(range_masked, None, 0, 1, cv2.NORM_MINMAX)
            plt.imshow(range_masked)
            plt.title('range_masked', fontsize=16)
            plt.axis('off')

            plt.subplot(6, 1, 5)
            range_pred = pred_dicts['range_pred'][:, :, 0]
            range_pred = cv2.normalize(range_pred, None, 0, 1, cv2.NORM_MINMAX)
            plt.imshow(range_pred)
            plt.title('reconstruction', fontsize=16)
            plt.axis('off')

            plt.subplot(6, 1, 6)
            range_paste = pred_dicts['range_paste'][:, :, 0]
            range_paste = cv2.normalize(range_paste, None, 0, 1, cv2.NORM_MINMAX)
            plt.imshow(range_paste)
            plt.title('reconstruction + visible', fontsize=16)
            plt.axis('off')

            file_name  = root_dir + 'RANGE_FULL_' + '%06d.png' % i
            plt.savefig(file_name)

            print('saving figures: ', i)


    # logger.info('**********************End evaluation %s/%s(%s)**********************' %
    #             (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
