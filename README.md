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

**Keywords**:  indoor localization, multi-view image, image recognition, similarity image search, GeM pooling

<div align="center">
<img src="https://user-images.githubusercontent.com/52001212/119213251-b833f280-baf8-11eb-9185-5690351df058.png" width="400px">　　<img src="https://user-images.githubusercontent.com/52001212/119213274-df8abf80-baf8-11eb-95d5-a0c3fa208b1b.png" width="300px">
</div>


## Datasets
- TUS Library Dataset: <br>
TUS Library Dataset is our proprietary dataset: it is a set of images taken at the Tokyo University of Science (TUS) Katsushika Campus Library (floor area: 3,358 m²). We captured reference images at 159 locations × 4 directions (636 images in total) taken at about 1[m] intervals by an iPhoneSE. Query images of 42 locations × 4 directions (168 images in total) were taken at random locations with an iPhone8Plus. All the images had size of 480×640[px].

<div align="center">
<img src="https://user-images.githubusercontent.com/52001212/119213393-a99a0b00-baf9-11eb-996a-d4de205c03c2.jpg" height="200px">
</div>

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
The percentage of query images where the distances between the estimated location and the ground truth location are within 1[m] is reported as One-Meter-Level Accuracy.

<img src="https://user-images.githubusercontent.com/52001212/119213493-6d1adf00-bafa-11eb-896a-ba12c0b590ac.jpg" height="200px">

## Conference
- 李 欣耘(東京理科大学)，古田 諒佑(東京大学)，入江 豪(NTTコミュニケーション科学基礎研究所)，谷口 行信(東京理科大学)，“多視点画像及びGeM Poolingを用いた屋内位置推定の高精度化”，画像電子学会 第295回研究会，2020．
- Xinyun Li, Ryosuke Furuta, Go Irie, and Yukinobu Taniguchi, "Accurate Indoor Localization Using Multi-View Image Distance", IEVC, 2021.
