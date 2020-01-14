# Deep Sort with PyTorch

This is a fork from [this repository](https://github.com/ZQPei/deep_sort_pytorch.git). Go to its main page for full documentation on it.

## Dependencies
- python 3 (python2 not sure)
- numpy
- scipy
- opencv-python
- sklearn
- torch >= 0.4
- torchvision >= 0.1
- pillow
- vizer
- edict

## Build instructions

**0. Check all dependencies installed**

```sh
pip install -r requirements.txt
```

**1. Clone this repository**

```sh
git clone git@github.com:tecnalia-advancedmanufacturing-robotics/deep_sort_pytorch.git
```

**2. Download YOLOv3 parameters**

```sh
cd detector/YOLOv3/weight/
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
cd ../../../
```

**3. Download deepsort parameters ckpt.t7**

```sh
cd deep_sort/deep/checkpoint
# download ckpt.t7 from https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6 to this folder
cd ../../../
```  

**4. Compile nms module**

```sh
cd detector/YOLOv3/nms
sh build.sh
cd ../../..
```

## References

- paper: [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)

- code: [nwojke/deep_sort](https://github.com/nwojke/deep_sort)

- paper: [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

- code: [Joseph Redmon/yolov3](https://pjreddie.com/darknet/yolo/)
