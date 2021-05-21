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
Experiments on two image datasets of real indoor scenes demonstrate the effectiveness of the proposed method.

**Keywords**:  indoor localization, multi-view image, image recognition, similarity image search

## Datasets
・TUS Library Dataset:

TUS Library Dataset is our proprietary dataset: it is a set of images taken at the Tokyo University of Science (TUS) Katsushika Campus Library (floor area: 3,358 m²). We captured reference images at 159 locations × 4 directions (636 images in total) taken at about 1[m] intervals by an iPhoneSE. Query images of 42 locations × 4 directions (168 images in total) were taken at random locations with an iPhone8Plus. All the images had size of 480×640[px].

・West Coast Plaza (WCP) dataset:

WCP Dataset is a public dataset of images taken at a shopping mall in Singapore (floor area: 15,000 m²). We have reference images of 316 locations × 4 directions (1264 images in total) taken at about 1[m] intervals with a Vivo Y79 and query images of 78 locations × 4 directions (312 images in total) were taken at random locations with a Vivo Y79.

## Installation
- Python 3.8.5
- PyTorch 1.8.0+cu111

