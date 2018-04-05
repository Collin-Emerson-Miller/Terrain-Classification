# Terrain-Classification
This is the repository for classifying terrain such as sand rock and gravel using machine learning.

To install all necessary packages run:
```
pip install -r requirements.txt
```

To train a model;

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

An example:
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