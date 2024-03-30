# Enhancing Wildfire Detection Accuracy from Satellite Imagery using Generative Algorithms

The aim of this dissertation is to develop and optimize a high-accuracy wildfire detection model using low-resolution satellite imagery for early mitigation and improved response efforts, specifically designed for resource-constrained environments.

## Key Components

### Models:

- UNET Model: Fire detection using a UNET architecture.
- GAN Model: Feature enhancement with a Generative Adversarial Network for improved detection in combination with the UNET model.

### Implementation:

- Language: Python
- Framework: TensorFlow

## Requirements

- TensorFlow
- OpenCV-Python

### Additional Libraries

- NumPy
- pandas
- Keras
- Matplotlib
- Tabulate


## Getting Started

### Clone:
```
git clone https://github.com/ayhameed/Fire-detection
```
### Install Dependencies:
```
pip install -r requirements.txt
```
### Usage
Detailed instructions are provided within the following folders:

- models/unet
- models/gan

### Folder structure
The project has the folllowing folders
- Fire-detection: this is the root folder 
    - /annotations : this contains the annotated boundaries of the fire in JSON
    - /checkpoints: contains the model checkpoint for the UNET and GAN.
    - /dataset: this consists of :
        - /images : contains the satellite imagery
        - /tiles : contains the normalized tiled images
        - /masks : contains the image masks for the tiles
    - /functions: this contains defined functions and the model scripts to create the GAN and UNET models
    - /model: contains the ipynb files for both models and the data preprocessing script

### Contact
hameedyunusa@outlook.com
