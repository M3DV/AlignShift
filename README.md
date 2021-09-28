# AlignShift

**NEW**: Code for our new MICCAI'21 paper "Asymmetric 3D Context Fusion for Universal Lesion Detection" will also be pushed to this repository soon.

*AlignShift*: Bridging the Gap of Imaging Thickness in 3D Anisotropic Volumes ([MICCAI'20, early accepted](https://arxiv.org/abs/2005.01969))

## Key contributions

* *AlignShift* aims at a **plug-and-play replacement** of standard 3D convolution for 3D medical images, which enables 2D-to-3D pretraining as [ACS Convolutions](https://github.com/M3DV/ACSConv/). It converts theoretically any 2D pretrained network into thickness-aware 3D network. 
* *AlignShift*  bridges the performance gap between thin- and thick-slice volumes by a unified framework. Remarkably, the *AlignShift*-converted networks behave like 3D for the thin-slice, nevertheless degenerate to 2D for the thick-slice adaptively.
* Without whistles and bells, we outperform previous state of the art by considerable margins on large-scale DeepLesion benchmark for universal lesion detection.

## Code structure

* ``alignshift``
  the core implementation of AlignShift convolution and TSM convolution, including the operators, models, and 2D-to-3D/AlignShift/TSM model converters. 
  * ``operators``: include AlignShiftConv, TSMConv.
  * ``converters.py``: include converters which convert 2D models to 3dConv/AlignShiftConv/TSMConv counterparts.
  * ``models``: Native AlignShift/TSM models. 
* ``deeplesion`` 
  the experiment code is base on [mmdetection](https://github.com/open-mmlab/mmdetection)
,this directory consists of compounents used in mmdetection.
* ``mmdet`` 
## Installation

 * git clone this repository
 * pip install -e . 
  
## Convert a 2D model into 3D with a single line of code

```python
from converter import Converter
import torchvision
from alignshift import AlignShiftConv
# m is a standard pytorch model
m = torchvision.models.resnet18(True)
alignshift_conv_cfg = dict(conv_type=AlignShiftConv, 
                          n_fold=8, 
                          alignshift=True, 
                          inplace=True,
                          ref_spacing=0.2, 
                          shift_padding_zero=True)
m = Converter(m, 
              alignshift_conv_cfg, 
              additional_forward_fts=['thickness'], 
              skip_first_conv=True, 
              first_conv_input_channles=1)
# after converted, m is using AlignShiftConv and capable of processing 3D volumes
x = torch.rand(batch_size, in_channels, D, H, W)
thickness = torch.rand(batch_size, 1)
out = m(x, thickness)
```

## Usage of AlignShiftConv/TSMConv operators

```python
from nn.operators import AlignShiftConv, TSMConv
x = torch.rand(batch_size, 3, D, H, W)
thickness = torch.rand(batch_size, 1)
# AlignShiftConv to process 3D volumnes
conv = AlignShiftConv(in_channels=3, out_channels=10, kernel_size=3, padding=1, n_fold=8, alignshift=True, ref_thickness=2.0)
out = conv(x, thickness)
# TSMConv to process 3D volumnes
conv = TSMConv(in_channels=3, out_channels=10, kernel_size=3, padding=1, n_fold=8, tsm=True)
out = conv(x)
```

## Usage of native  AlignShiftConv/TSMConv models

```python
from nn.models import DenseNetCustomTrunc3dAlign, DenseNetCustomTrunc3dTSM
net = DenseNetCustomTrunc3dAlign(num_classes=3)
B, C_in, D, H, W = (1, 3, 7, 256, 256)
input_3d = torch.rand(B, C_in, D, H, W)
thickness = torch.rand(batch_size, 1)
output_3d = net(input_3d, thickness)
```

## How to run the experiments

* Dataset

  * Download [Deeplesion dataset](https://nihcc.box.com/v/DeepLesion)
  * Before training, mask should be generated from bounding box and recists. [mask generation](./deeplesion/dataset/generate_mask_with_grabcut.md)

* Preparing mmdetection script

  * Specify input ct slices in [./deeplesion/mconfigs/densenet_align.py](./deeplesion/mconfigs/densenet_align.py) through modifing NUM_SLICES in dict dataset_transform
  
  * Specify data root in [./deeplesion/ENVIRON.py](./deeplesion/ENVIRON.py)
  
* Model weights

  Our trained weights published on: 
   * BaiDuYun:链接: https://pan.baidu.com/s/1NsCkvjZdAgi9navg3_ry3g 提取码: h2wc
   * Google Drive: https://drive.google.com/drive/folders/1_ApYs5vb_VzkdqK02lb861Psj-GSdznV?usp=sharing

* Training
  ```bash
  ./deeplesion/train_dist.sh ${mmdetection script} ${dist training GPUS}
  ```

  * Train AlignShiftConv models 
  ```bash
  ./deeplesion/train_dist.sh ./deeplesion/mconfig/densenet_align.py 2
  ```

  * Train TSMConv models 
  ```bash
  ./deeplesion/train_dist.sh ./deeplesion/mconfig/densenet_tsm.py 2
  ```
 * Evaluation 
   ```bash
   ./deeplesion/eval.sh ${mmdetection script} ${checkpoint path}
      ```
   ```bash
   ./deeplesion/eval.sh ./deeplesion/mconfig/densenet_align.py ./deeplesion/model_weights/alignshift_7slice.pth
   ```
