# MMdetection v2 with mosaic data augmentation

## Introduction
### add yolov5's data aug to mmdetv2.  
Support SOTA model DetectoRS    
Please check configs/wheat
Please check mmdet/datasets/mosaic.py mosaiccoco.py my_mosaic.py  
Please check mmdet/datasets/pipelines/loading  


## Useage
Replace LoadImageFromFile and LoadAnnotations with LoadMosaicImageAndAnnotations in train_pipeline

# Reference 
### mmdetection
https://github.com/open-mmlab/mmdetection
### DetectoRS
https://github.com/joe-siyuan-qiao/DetectoRS
### Mosaic data augmentation
https://github.com/ultralytics/yolov5
