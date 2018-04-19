# Terrain-Classification

Terrain-Classification is an open-source 3-D point cloud/terrain classification application.  
Terrain-Classification uses RGB-D images from the X-Box Kinect to generate a point cloud that
contains terrain classifications for each point.

## Installation

To install all necessary packages run:
```
pip install -r requirements.txt
```

## Usage
This section details how to set up the terrain classification application.


### Training a model
To train a model open a terminal and run

```
python train_model.py <model_name> <corpus_path>
```

where the directory tree is:

```
-   <corpus_path>

    -   <class_folder_2>

        -   <image_1.jpg>
        -   <image_2.jpg>
        -   ...
        -   <image_n.jpg>

    -   <class_folder_2>

        -   <image_1.jpg>
        -   <image_2.jpg>
        -   ...
        -   <image_n.jpg>

    -   ...
    -   <class_folder_n>

        -   <image_1.jpg>
        -   <image_2.jpg>
        -   ...
        -   <image_n.jpg>
```

#### Example
```
python train_model.py inceptionV3 terrain_images
```

where the directory tree is:

```
-   terrain_images

    -   concrete

        -   image_1.jpg
        -   image_2.jpg
        -   ...
        -   image_n.jpg

    -   grass

        -   image_1.jpg
        -   image_2.jpg
        -   ...
        -   image_n.jpg

    -   ...
    -   rock

        -   image_1.jpg
        -   image_2.jpg
        -   ...
        -   image_n.jpg
```
