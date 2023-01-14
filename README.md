# Automated Weed Detection and Removal in Agricultural Fields

The project is based on the Intel® oneAPI Hackathon for Open Innovation Hackathon 

## Computer Vision Challenge Track: Target and Eliminate

### Problem:

Weeds to increase crop yields Weeds are an unwanted intruder in the agricultural business. They steal nutrients, water, land, and other critical resources to grow healthy crops. These intruders can lead to lower yields and inefficient deployment of resources by farmers. One known approach is to use pesticides to remove weeds, but aggressive pesticides create health risks for humans. Computer vision technology can automatically detect the presence of weeds and use targeted remediation techniques to remove them from fields with minimal environmental impact.

### Expected Solution:

In this hackathon track, you will be tasked with training and deploying a model into a simulated production environment - where your binary-classification accuracy (F1 score) and inference time will be used to rank you against other teams competing for this track's top spot.

## Objective

The objective of this project is to develop a computer vision model that can accurately detect and remove weeds in agricultural fields with minimal environmental impact. The proposed approach uses a Residual Network (ResNet) architecture with Auto Mixed Precision (AMP) to classify images of crops and weeds. The use of AMP allows for dynamic adjustment of the precision of computations during training and inference, which can improve the performance of the model. The use of oneAPI Analytical Toolkit will enhance the performance and efficiency of the model, leading to improved crop yields and more efficient use of resources.

## Implementation

The implementation of this project will involve the following steps:
1. Collect and label a dataset of images of crops and weeds from real-world environments.
2. Use the oneAPI Data Analytics Library (DAL) to preprocess the dataset and prepare it for training.
3. Train a ResNet with AMP to classify images as either crops or weeds.
4. Optimize the model's performance using the oneAPI Math Kernel Library (MKL) and oneAPI Graph Analytics Library (GAL)
5. Use the oneAPI Model Analyzer and Debugger (MAD) to analyze the model's performance and identify any areas for  improvement.
6. Deploy the trained model to a simulated production environment using the oneAPI Deployment Manager.
7. Use the oneAPI Performance Profiler to monitor the model's runtime performance and identify any bottlenecks that may be impacting its performance.
8. Continuously monitor the model's performance and make updates as needed to improve its accuracy.


## Requirements

    • TensorFlow or PyTorch deep learning framework in Anaconda IDE

    • oneAPI Analyticals toolkit

    • OpenCV

    • Python >= 3.7

    • NumPy

    • Pandas

    • Matplotlib




## Installation
1.	Install the deep learning framework by following the instructions on the official website ([TensorFlow](https://www.intel.com/content/www/us/en/developer/articles/guide/optimization-for-tensorflow-installation-guide.html) or [PyTorch](https://www.intel.com/content/www/us/en/developer/articles/guide/getting-started-with-intel-optimization-of-pytorch.html))
2.	Install oneAPI toolkit by following the instructions on the official website
3.	Install OpenCV and other required python libraries using pip:

    Copy the code
    ```
    pip install opencv-python pip install numpy pip install pandas pip install matplotlib 
    ```

## Usage
1.	Clone the repository:
Copy code
git clone https://github.com/Nitin-Mane/Automated-Weed-Detection.git 
2.	Download the dataset and place it in the project folder.
3.	Use the oneAPI DAL to preprocess the dataset
4.	Train the model using the deep learning framework with AMP and oneAPI DNNL
5.	Optimize the model's performance using oneAPI MKL and GAL
6.	Analyze the model's performance using oneAPI MAD
7.	Deploy the model to the simulated production environment using oneAPI Deployment Manager
8.	Monitor the model's performance using oneAPI Performance Profiler


# License

This project is licensed under the MIT License - see the LICENSE file for details.
