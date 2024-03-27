This project aims to simulate fire detection in space in near real time leveraging "Space 2.0" methods and consists of two parts: 

## The Model
The project includes two models for fire detection:

1. UNET Model: This model is designed to detect fire using the UNET architecture.
2. GAN Model: The GAN (Generative Adversarial Network) is used to enhance features, which are then passed through the UNET model again for improved performance.

The inference process is implemented in Python using TensorFlow.

### General Requirements
To run this project, you will need the following:

- JavaScript
- Node.js
- TensorFlow
- OpenCV-Python

This project utilizes an ecosystem of libraries alongside TensorFlow. These libraries are:

- NumPy: Provides fundamental numerical operations and data structures for efficient array manipulation and streamlining data processing.
- pandas: Enables efficient data loading, cleaning, and manipulation, facilitating effective pre-processing steps.
- Keras: Offers a high-level API on top of TensorFlow, simplifying the creation and training of neural networks, and making the development process more manageable.
- OpenCV: Provides image processing and computer vision functionalities essential for pre-processing satellite imagery and visualizing results.
- Matplotlib: Creates insightful and informative visualizations to understand data, model performance, and predictions, aiding in model evaluation and interpretation.
- Ray: A distributed computing framework enables parallelization of training across multiple machines or GPUs, significantly accelerating the training process.
- Shapely: Used for creating and handling polygons.

Please refer to the readme files in the respective folders for detailed instructions on how to use this project.
