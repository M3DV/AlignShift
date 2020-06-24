import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from pycocotools import mask as mutils
import sys
# sys.path.append('add your workspace here')
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
# from dataset import DeepLesionDataset
import cv2
import random
import matplotlib.pyplot as plt
from mmcv import Config
import mmcv
from mmcv.runner import get_dist_info, load_checkpoint
from mmcv.parallel import MMDataParallel
import random
import pickle
import gc
import argparse
from mmdet.datasets.registry import DATASETS, PIPELINES
from mmdet.models.registry import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
                       ROI_EXTRACTORS, SHARED_HEADS)
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox, force_fp32,
                        multi_apply, multiclass_nms)
# from deeplesion.evaluation.visualize import draw_bounding_boxes_on_image_array
from deeplesion.evaluation.evaluation_metrics import sens_at_FP

def parse_args():
    parser = argparse.ArgumentParser(description='eval deeplesion')
    # parser.add_argument('config', help='train config file path')
    parser.add_argument('--config', help='config path')
    parser.add_argument('--checkpoint', help='checkpoint path')
    args = parser.parse_args()

    return args

def generate_cfg(checkpoint):    
    d = torch.load(checkpoint, map_location=torch.device('cpu'))
    cfg_path = checkpoint.replace('.pth','.py')
    if not os.path.exists(cfg_path):
        with open(cfg_path,'w') as f:
            f.write(d['meta']['config'])
    return cfg_path

def get_model(cfg_path):
    cfg = Config.fromfile(cfg_path)
    cfg.data.imgs_per_gpu=1
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.data.val.test_mode = True
    # cfg.test_cfg.rcnn.score_thr=0.001
    #build dataset
    print(cfg.description)
    dataset = build_dataset(cfg.data.test)
    data_loadertest = build_dataloader(
                                        dataset,
                                        imgs_per_gpu=1,
                                        workers_per_gpu=0,
                                        dist=False,
                                        shuffle=False)

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    model = MMDataParallel(model, device_ids=[0])
    model.CLASSES = dataset.CLASSES
    return model, data_loadertest

def single_gpu_test(model, data_loader):
    model.eval()
    results = []
#     dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(data_loader))
    with torch.no_grad():
        for i, data in enumerate(data_loader):

            gt_boxes = data.pop('gt_bboxes')
            r = model(return_loss=False, rescale=False, **data)   
            # inference_time.append(time.time() - start_time)
            data['gt_boxes'] = gt_boxes
            data['bboxes'] = r[0]
            data['segs'] = r[1]
            # data['img'] = data['img'].data[0][data['img'].data[0].shape[0]//2]
            data.pop('img')
            results.append(data)
            prog_bar.update()
    return results

def write_metrics(outputs, log_path, epoch):
    avgFP = [0.5, 1, 2, 4, 8, 16]
    iou_th = 0.5
    s1_box=[]
    s1_gt=[]
    s5_box=[]
    s5_gt=[]
    so_box=[]
    so_gt=[]
    for d in outputs:
        if d['thickness'].data[0]<=2.:
            s1_box.append(np.vstack(d['bboxes']))
            s1_gt.append(d['gt_boxes'].data[0][0].numpy())
        elif d['thickness'].data[0]==5.:
            s5_box.append(np.vstack(d['bboxes']))
            s5_gt.append(d['gt_boxes'].data[0][0].numpy()) 
        else:
            so_box.append(np.vstack(d['bboxes']))
            so_gt.append(d['gt_boxes'].data[0][0].numpy()) 
            
    sens1 = sens_at_FP(s1_box, s1_gt, avgFP, iou_th)
    sens2 = sens_at_FP(s5_box, s5_gt, avgFP, iou_th)
    sens = sens_at_FP(s1_box+s5_box+so_box, s1_gt+s5_gt+so_gt, avgFP, iou_th)
    s = str(epoch)+':\t'+str(sens)+'\t'+str(sens1)+'\t'+str(sens2)+'\t align srl\n'
    print(s)
    with open(log_path,'a+') as f:
        f.write(s)
    return s 

if __name__ =='__main__':
    # checkpoint_path = f'/mnt/data3/deeplesion/dl/work_dirs/densenet_3d_acs_r2/latest.pth'
    args = parse_args()
    checkpoint = args.checkpoint
    cfg_path = args.config#generate_cfg(checkpoint)
    model, dl = get_model(cfg_path)
    # log_path = checkpoint.replace('latest.pth', 'metrix_log.txt')
    log_path = '/mnt/data3/alignconv/logs/metrix_log.txt'
    load_checkpoint(model, checkpoint, map_location='cpu', strict=True)
    outputs = single_gpu_test(model, dl)
    r = write_metrics(outputs, log_path, 'N/A')
    print(r)
