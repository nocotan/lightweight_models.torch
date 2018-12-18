# PyTorch Implementation for Lightweight Models

## Dependencies

```bash
torch==1.0.0
torchvision==0.2.1
```

## Models

* SqueezeNet
  - SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size>
  - [arxiv](https://arxiv.org/abs/1602.07360)
* MobileNet
  - MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
  - [arxiv](https://arxiv.org/abs/1704.04861)
* MobileNet v2
  - MobileNetV2: Inverted Residuals and Linear Bottlenecks
  - [arxiv](https://arxiv.org/abs/1801.04381)
* ShuffleNet
  - ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
  - [arxiv](https://arxiv.org/abs/1707.01083)
* ShuffleNet v2
  - ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
  - [arxiv](https://arxiv.org/abs/1807.11164)

## Usage

```bash
usage: main.py [-h] [--mode MODE] [--model MODEL] [--dataset DATASET]
               [--dataroot DATAROOT] [--batch_size BATCH_SIZE]
               [--n_epochs N_EPOCHS] [--lr LR] [--n_gpus N_GPUS]
               [--checkpoint CHECKPOINT] [--pretrained PRETRAINED]

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE
  --model MODEL
  --dataset DATASET
  --dataroot DATAROOT
  --batch_size BATCH_SIZE
  --n_epochs N_EPOCHS
  --lr LR
  --n_gpus N_GPUS
  --checkpoint CHECKPOINT
  --pretrained PRETRAINED
```
