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


## Getting Started

### Clone:
Clone the repository by typing the command below in your CLI
```
git clone https://github.com/ayhameed/Fire-detection
```
### Install Dependencies:
Install the libraries needed to replicate this, by runnning the command below
```
pip install -r requirements.txt
```
### Usage
Detailed instructions are provided within the following folders:
- model/image_preprocessing.ipynb
- model/unet.ipynb
- model/unet-with-gan-data
- model/gan_mse_loss.ipynb

### Folder structure
The project has the folllowing folders
- /Fire-detection: this is the root folder 

    - /annotations ** : this contains the annotated boundaries of the fire in JSON

    - /checkpoints **: contains the model checkpoint for the UNET and GAN.
        - unet_checkpoint.keras : the checkpoint for the UNET trained without gan augumentation
        - unet_with_gan_checkpoint.kera : the checkpoint for the UNET trained with gan augumentation
        - generator_MSE_Sigmoid_v4.keras : the checkpoint for the generator used to create the synthetic images 

    - /dataset: contains the dataset used to train and test the model and consists of :
        - /images : contains the satellite imagery
        - /tiles : contains the tiled images
        - /masks : contains the binary masks for the tiles
        - /synthesized_tiles : contains the tiles generated using GAN
        - /synthesized_masks : contains the binary masks for the tiles generated using GAN
    - /functions: this contains defined functions and the model scripts to create the GAN and UNET models
        - /gan : contains functions used to build and compile the GAN model

    - /model: contains the ipynb files for both models and the data preprocessing script
        - load_binary_mask.py : a py script to load binary masks from a path 
        - unet : the notebook file containing the unet model and results
        - gan : the notebook file containing the data generation process
        - unet_gan : the notenook file containing the unet model trained with gan data and the results 

    - requirements.txt : a file containing all the dependencies needed for this project to run
    - readme.md : a file detailing how to replicate the project

### Note:
- dir or files with ** appended to the name have been added to the gitignore file 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ayhameed/Fire-detection/HEAD)