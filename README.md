# Accurate Indoor Localization Using Multi-View Image Distance

Due to the increasing complexity of indoor facilities such as shopping malls and train stations, 
there is a need for a technology that can find the current location of a user using a smartphone or other devices, 
even in indoor areas where GPS signals cannot be received. 
Indoor localization methods based on image recognition have been proposed as solutions. 
While many localization methods have been proposed for outdoor use, 
indoor localization has difficultly in achieving high accuracy from just one image taken by the user (query image), 
because there are many similar objects (walls, desks, etc.) and there are only a few cues that can be used for localization. 
In this paper, we propose a novel indoor localization method that uses multi-view images. 
The basic idea is to improve the localization quality by retrieving the pre-captured image with location information (reference image) that best matches the multi-view query image taken from multiple directions around the user. 
To this end, we introduce a simple metric to evaluate the distance between multi-view images. 

**Keywords**:  indoor localization, multi-view image, image recognition, similarity image search, GeM pooling[1]

<div align="center">
<img src="https://user-images.githubusercontent.com/52001212/119598331-b5e5d700-be1d-11eb-8390-187fe17da6fe.jpg" width="500px">　　<img src="https://user-images.githubusercontent.com/52001212/119598367-cb5b0100-be1d-11eb-8b1c-d04350c27f89.jpg" width="300px">
</div>

## Multi-view image distace
<img src="<a href="https://www.codecogs.com/eqnedit.php?latex={\rm&space;distance}({\mathbf&space;Q},&space;{\mathbf&space;R}_j)&space;=&space;\displaystyle&space;\min_{\sigma&space;\in&space;S_4}&space;\sum^4_{a=1}{\rm&space;dist}(Q_{a},&space;R_{j\sigma_a})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{\rm&space;distance}({\mathbf&space;Q},&space;{\mathbf&space;R}_j)&space;=&space;\displaystyle&space;\min_{\sigma&space;\in&space;S_4}&space;\sum^4_{a=1}{\rm&space;dist}(Q_{a},&space;R_{j\sigma_a})" title="{\rm distance}({\mathbf Q}, {\mathbf R}_j) = \displaystyle \min_{\sigma \in S_4} \sum^4_{a=1}{\rm dist}(Q_{a}, R_{j\sigma_a})" /></a>" />

## Datasets
- TUS Library Dataset: <br>
　TUS Library Dataset is our proprietary dataset: it is a set of images taken at the Tokyo University of Science (TUS) Katsushika Campus Library (floor area: 3,358 m²). We captured reference images at 159 locations × 4 directions (636 images in total) taken at about 1[m] intervals by an iPhoneSE. Query images of 42 locations × 4 directions (168 images in total) were taken at random locations with an iPhone8Plus. All the images had size of 480×640[px]. <br>
　You can download it from [here](https://drive.google.com/drive/folders/1pPIgqWh0kEy-_kt5TllEmGhuzAFtn95X?usp=sharing). Put the image data under `dataset/library/`.

![dataset](https://user-images.githubusercontent.com/52001212/119600510-268ef280-be22-11eb-9cbd-c85fcfd95da0.jpg)

## Installation
- Python 3.8.5
- PyTorch 1.8.0+cu111

## Network
- ResNet152 (trained on google-landmarks-2018)

## Execution
```
python multi_library.py
```

## Result
Evaluation Metrics: <br>
　The percentage of query images where the distances between the estimated location and the ground truth location are within 1[m] is reported as One-Meter-Level Accuracy.

<img src="https://user-images.githubusercontent.com/52001212/119213493-6d1adf00-bafa-11eb-896a-ba12c0b590ac.jpg" height="200px">

## Reference
[1] Filip Radenović, et al. Fine-Tuning CNN Image Retrieval with No Human Annotation. *TPAMI*, Vol. 41, No. 7, pp. 1655–1668, 2019.

## Conference
- Xinyun Li (Tokyo University of Science)，Ryosuke Furuta (The University of Tokyo)，Go Irie (NTT Communication Science Laboratories)，and Yukinobu Taniguchi (Tokyo University of Science)，“Accurate Indoor Localization Using Multi-View Images and Generalized Mean Pooling”，*IIEEJ*，2020．
- Xinyun Li, Ryosuke Furuta, Go Irie, and Yukinobu Taniguchi, "Accurate Indoor Localization Using Multi-View Image Distance", *IEVC*, 2021.
