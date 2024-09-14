## Introduction

### A Irregular-patch Graph Attention Network for Underwater Object Detection.

Underwater object detection is an important computer vision task that has been widely used in marine life identification and tracking. However, there are a series of challenges such as low contrast condition, occlusion condition, unbalanced light condition and small dense objects in underwater object detection.
Attention mechanism has been proven powerful in feature extraction. However, attention mechanisms usually divide image into fixed patches. This methods lead to the splitting up of continuous structures, which hinders the use of similar information in other areas to enhance image details. In fact, using graph structure is more flexible and effective for visual perception, since it can capture complex target relation and context information effectively.
Thus, we apply graph attention mechanisms to irregular patches and propose an Irregular-patch Graph Attention Network (IGA-Net). Firstly, the superpixel segmentation method is used to segment the image to reduce noise. Secondly, the global graph and local graph are constructed to obtain internal structures. Finally, to handle occlusion and small objects, a distinctive feature three-way handshake (F3H) module is proposed to fuse information from global and local graph. To demonstrate the effectiveness of the proposed method, we conduct comprehensive evaluations on six challenging underwater datasets UTDAC2020, RUOD, Brackish, TrashCan and WPBB. Experimental results demonstrate that the proposed IGA-Net achieves superior performance on six challenging underwater datasets.

![image](https://github.com/user-attachments/assets/239f89d0-223c-4669-ad6b-75a9428df089)


![image](https://github.com/user-attachments/assets/e997ada2-96bc-4ee4-9f6a-b93832020501)



### Download Dataset

#### (1) UTDAC2020 dataset (https://drive.google.com/file/d/1avyB-ht3VxNERHpAwNTuBRFOxiXDMczI/view?usp=sharing)

#### (2) RUOD dataset (https://github.com/dlut-dimt/RUOD)

#### (3) Trashcan dataset (https://conservancy.umn.edu/handle/11299/214865)

#### (4) WPBB dataset (https://github.com/fedezocco/MoreEffEffDetsAndWPBB-TensorFlow/tree/main/WPBB_dataset)

#### (5) Brackish dataset (https://www.kaggle.com/datasets/aalborguniversity/brackish-dataset)



### Some Results

![image](https://github.com/user-attachments/assets/268a5187-fb3f-4aa5-b340-e2e7c6e47f4d)
![image](https://github.com/user-attachments/assets/9ee59ec3-92db-4076-a6f5-2e1bebaaeb8c)



## Dependencies

- Python==3.9.6
- PyTorch==1.10.0
- mmdetection==3.2.0
- mmcv==2.1.0
- numpy==1.26.2

## Installation

The basic installation follows with [mmdetection](https://github.com/mousecpn/mmdetection/blob/master/docs/get_started.md). It is recommended to install locally. 

