# Introduction
This is an official PyTorch implementation of our PRCV 2023 paper [Mobile-LRPose: Low-Resolution Representation Learning for Human Pose Estimation in mobile devices](https://pages.github.com/). We construct a network backbone based
on the modified MobileNetV2 to only generate low-resolution representations. Then, to enhance the capability of keypoints localization for our
model, we also make crucial improvements consisting of bottleneck atrous
spatial pyramid, local-space attention, coordinate attention and position
embedding. In addition, we design two different network heads for 2D
and 3D pose estimation to explore the extensibility of the backbone.
Our model achieves superior performance to state-of-the-art lightweight
2D pose estimation models on both COCO and MPII datasets, which
achieves 25+ FPS on HUWEI Kirin 9000 and outperforms MoveNet in
the same device. Our 3D model also makes nearly 50% and 90% reduction
on parameters and FLOPs compared to lightweight alternatives. 
# Results
Results on COCO val2017 with detector having human AP of 68.4 on COCO val2017 dataset
| Input size | #Params | GFLOPs | AP | AP^50 | AP^75 | AR |
| -----------| ------- | ------ | -- | ----- | ----- | -- |
|   256×192  |   1.5M  |  0.29  |68.4|  90.5 |  76.0 |71.8|
Results on MPII val dataset
| Input size | #Params | GFLOPs | PCKh |
| -----------| ------- | ------ | ---- |
|   256×256  |   1.5M  |  0.39  | 87.5 |
Results on Human3.6M protocol1 dataset
| Dir. | Dis. | Eat. | Gre. | Phon. | Pose | Pur. | Sit. | SitD. | Smo. | Phot. | Wait | Walk | WalkD. | WalkP. | Avg. | Params. | GFLOPs |
| ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ----- | ---- | ----- | ---- | ---- | ------ | ----- | ---- | ------- | ------ |
| 37.1 | 37.5 | 46.6 | 41.7 | 40.0 | 37.6 | 36.0 | 41.1 | 53.3 | 43.4 | 48.5 | 36.7 | 30.0 | 42.5 | 36.2 | 40.8 | 1.86M | 0.45 |
Results on Human3.6M protocol2 dataset
| Dir. | Dis. | Eat. | Gre. | Phon. | Pose | Pur. | Sit. | SitD. | Smo. | Phot. | Wait | Walk | WalkD. | WalkP. | Avg. | Params. | GFLOPs |
| ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ----- | ---- | ----- | ---- | ---- | ------ | ----- | ---- | ------- | ------ |
| 51.5 | 60.5 | 57.5 | 55.8 | 62.0 | 52.6 | 53.9 | 73.9 | 87.1 | 60.9 | 65.2 | 54.5 | 46.0 | 60.9 | 53.8 | 60.42 | 1.86M | 0.45 |

