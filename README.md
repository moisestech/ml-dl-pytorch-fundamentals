# dl-pytorch

Deep Learning Pytorch

# Deep Learning (PyTorch) - ND101 v7

This repository contains material for [Deep Learning Fundamentals](https://www.udacity.com/course/deep-learning-nanodegree--nd101). It consists of a bunch of tutorial notebooks for various deep learning topics. In most cases, the notebooks lead you through implementing models such as convolutional networks, recurrent networks, and GANs. There are other topics covered such as weight initialization and batch normalization.

## Table Of Contents

### Tutorials

### Introduction to Neural Networks

- [Introduction to Neural Networks](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/intro-neural-networks): Learn how to implement gradient descent and apply it to predicting patterns in student admissions data.
- [Sentiment Analysis with NumPy](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/sentiment-analysis-network): [Andrew Trask](http://iamtrask.github.io/) leads you through building a sentiment analysis model, predicting if some text is positive or negative.
- [Introduction to PyTorch](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/intro-to-pytorch): Learn how to build neural networks in PyTorch and use pre-trained networks for state-of-the-art image classifiers.

### Convolutional Neural Networks

- [Convolutional Neural Networks](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/convolutional-neural-networks): Visualize the output of layers that make up a CNN. Learn how to define and train a CNN for classifying [MNIST data](https://en.wikipedia.org/wiki/MNIST_database), a handwritten digit database that is notorious in the fields of machine and deep learning. Also, define and train a CNN for classifying images in the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
- [Transfer Learning](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/transfer-learning). In practice, most people don't train their own networks on huge datasets; they use **pre-trained** networks such as VGGnet. Here you'll use VGGnet to help classify images of flowers without training an end-to-end network from scratch.
- [Weight Initialization](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/weight-initialization): Explore how initializing network weights affects performance.
- [Autoencoders](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/autoencoder): Build models for image compression and de-noising, using feedforward and convolutional networks in PyTorch.
- [Style Transfer](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/style-transfer): Extract style and content features from images, using a pre-trained network. Implement style transfer according to the paper, [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) by Gatys et. al. Define appropriate losses for iteratively creating a target, style-transferred image of your own design!

### Recurrent Neural Networks

- [Intro to Recurrent Networks (Time series & Character-level RNN)](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/recurrent-neural-networks): Recurrent neural networks are able to use information about the sequence of data, such as the sequence of characters in text; learn how to implement these in PyTorch for a variety of tasks.
- [Embeddings (Word2Vec)](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/word2vec-embeddings): Implement the Word2Vec model to find semantic representations of words for use in natural language processing.
- [Sentiment Analysis RNN](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/sentiment-rnn): Implement a recurrent neural network that can predict if the text of a moview review is positive or negative.
- [Attention](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/attention): Implement attention and apply it to annotation vectors.

### Generative Adversarial Networks

- [Generative Adversarial Network on MNIST](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/gan-mnist): Train a simple generative adversarial network on the MNIST dataset.
- [Batch Normalization](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/batch-norm): Learn how to improve training rates and network stability with batch normalizations.
- [Deep Convolutional GAN (DCGAN)](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/dcgan-svhn): Implement a DCGAN to generate new images based on the Street View House Numbers (SVHN) dataset.
- [CycleGAN](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/cycle-gan): Implement a CycleGAN that is designed to learn from unpaired and unlabeled data; use trained generators to transform images from summer to winter and vice versa.

### Deploying a Model (with AWS SageMaker)

- [All exercise and project notebooks](https://github.com/udacity/sagemaker-deployment) for the lessons on model deployment can be found in the linked, Github repo. Learn to deploy pre-trained models using AWS SageMaker.

### Projects

- [Predicting Bike-Sharing Patterns](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-bikesharing): Implement a neural network in NumPy to predict bike rentals.
- [Dog Breed Classifier](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-dog-classification): Build a convolutional neural network with PyTorch to classify any image (even an image of a face) as a specific dog breed.
- [TV Script Generation](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-tv-script-generation): Train a recurrent neural network to generate scripts in the style of dialogue from Seinfeld.
- [Face Generation](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-face-generation): Use a DCGAN on the CelebA dataset to generate images of new and realistic human faces.

### Elective Material

- [Intro to TensorFlow](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/tensorflow/intro-to-tensorflow): Starting building neural networks with TensorFlow.
- [Keras](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/keras): Learn to build neural networks and convolutional neural networks with Keras.

---

# Dependencies

## Configure and Manage Your Environment with Anaconda

Per the Anaconda [docs](http://conda.pydata.org/docs):

> Conda is an open source package management system and environment management system
> for installing multiple versions of software packages and their dependencies and
> switching easily between them. It works on Linux, OS X and Windows, and was created
> for Python programs but can package and distribute any software.

## Overview

Using Anaconda consists of the following:

1. Install [`miniconda`](http://conda.pydata.org/miniconda.html) on your computer, by selecting the latest Python version for your operating system. If you already have `conda` or `miniconda` installed, you should be able to skip this step and move on to step 2.
2. Create and activate \* a new `conda` [environment](http://conda.pydata.org/docs/using/envs.html).

\* Each time you wish to work on any exercises, activate your `conda` environment!

---

## 1. Installation

**Download** the latest version of `miniconda` that matches your system.

|        | Linux                            | Mac                              | Windows                         |
| ------ | -------------------------------- | -------------------------------- | ------------------------------- |
| 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64] |
| 32-bit | [32-bit (bash installer)][lin32] |                                  | [32-bit (exe installer)][win32] |

[win64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
[win32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
[mac64]: https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
[lin64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
[lin32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh

**Install** [miniconda](http://conda.pydata.org/miniconda.html) on your machine. Detailed instructions:

- **Linux:** https://conda.io/projects/conda/en/latest/user-guide/install/linux.html
- **Mac:** https://conda.io/projects/conda/en/latest/user-guide/install/macos.html
- **Windows:** https://conda.io/projects/conda/en/latest/user-guide/install/windows.html

## 2. Create and Activate the Environment

For Windows users, these following commands need to be executed from the **Anaconda prompt** as opposed to a Windows terminal window. For Mac, a normal terminal window will work.

#### Git and version control

These instructions also assume you have `git` installed for working with Github from a terminal window, but if you do not, you can download that first with the command:

```
conda install git
```

If you'd like to learn more about version control and using `git` from the command line, take a look at our [free course: Version Control with Git](https://www.udacity.com/course/version-control-with-git--ud123).

**Now, we're ready to create our local environment!**

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.

```
git clone https://github.com/udacity/deep-learning-v2-pytorch.git
cd deep-learning-v2-pytorch
```

2. Create (and activate) a new environment, named `deep-learning` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

   - **Linux** or **Mac**:

   ```
   conda create -n deep-learning python=3.6
   source activate deep-learning
   ```

   - **Windows**:

   ```
   conda create --name deep-learning python=3.6
   activate deep-learning
   ```

   At this point your command line should look something like: `(deep-learning) <User>:deep-learning-v2-pytorch <user>$`. The `(deep-learning)` indicates that your environment has been activated, and you can proceed with further package installations.

3. Install PyTorch and torchvision; this should install the latest version of PyTorch.

   - **Linux** or **Mac**:

   ```
   conda install pytorch torchvision -c pytorch
   ```

   - **Windows**:

   ```
   conda install pytorch -c pytorch
   pip install torchvision
   ```

4. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).

```
pip install -r requirements.txt
```

7. That's it!

Now most of the `deep-learning` libraries are available to you. Very occasionally, you will see a repository with an addition requirements file, which exists should you want to use TensorFlow and Keras, for example. In this case, you're encouraged to install another library to your existing environment, or create a new environment for a specific project.

Now, assuming your `deep-learning` environment is still activated, you can navigate to the main repo and start looking at the notebooks:

```
cd
cd deep-learning-v2-pytorch
jupyter notebook
```

To exit the environment when you have completed your work session, simply close the terminal window.

---

1 INTRO TO DEEP LEARNING

Deep Learning V2 PyTorch

Predicting Bike-Sharing-Dataset
Day.csv Hours.csv

==========================================
Bike Sharing Dataset
==========================================

Hadi Fanaee-T

Laboratory of Artificial Intelligence and Decision Support (LIAAD), University of Porto
INESC Porto, Campus da FEUP
Rua Dr. Roberto Frias, 378
4200 - 465 Porto, Portugal

=========================================
Background
=========================================

Bike sharing systems are new generation of traditional bike rentals where whole process from membership, rental and return
back has become automatic. Through these systems, user is able to easily rent a bike from a particular position and return
back at another position. Currently, there are about over 500 bike-sharing programs around the world which is composed of
over 500 thousands bicycles. Today, there exists great interest in these systems due to their important role in traffic,
environmental and health issues.

Apart from interesting real world applications of bike sharing systems, the characteristics of data being generated by
these systems make them attractive for the research. Opposed to other transport services such as bus or subway, the duration
of travel, departure and arrival position is explicitly recorded in these systems. This feature turns bike sharing system into
a virtual sensor network that can be used for sensing mobility in the city. Hence, it is expected that most of important
events in the city could be detected via monitoring these data.

=========================================
Data Set
=========================================
Bike-sharing rental process is highly correlated to the environmental and seasonal settings. For instance, weather conditions,
precipitation, day of week, season, hour of the day, etc. can affect the rental behaviors. The core data set is related to  
the two-year historical log corresponding to years 2011 and 2012 from Capital Bikeshare system, Washington D.C., USA which is
publicly available in http://capitalbikeshare.com/system-data. We aggregated the data on two hourly and daily basis and then
extracted and added the corresponding weather and seasonal information. Weather information are extracted from http://www.freemeteo.com.

=========================================
Associated tasks
=========================================

    - Regression:
    	Predication of bike rental count hourly or daily based on the environmental and seasonal settings.

    - Event and Anomaly Detection:
    	Count of rented bikes are also correlated to some events in the town which easily are traceable via search engines.
    	For instance, query like "2012-10-30 washington d.c." in Google returns related results to Hurricane Sandy. Some of the important events are
    	identified in [1]. Therefore the data can be used for validation of anomaly or event detection algorithms as well.

=========================================
Files
=========================================

    - Readme.txt
    - hour.csv : bike sharing counts aggregated on hourly basis. Records: 17379 hours
    - day.csv - bike sharing counts aggregated on daily basis. Records: 731 days

=========================================
Dataset characteristics
=========================================
Both hour.csv and day.csv have the following fields, except hr which is not available in day.csv - instant: record index - dteday : date - season : season (1:springer, 2:summer, 3:fall, 4:winter) - yr : year (0: 2011, 1:2012) - mnth : month ( 1 to 12) - hr : hour (0 to 23) - holiday : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule) - weekday : day of the week - workingday : if day is neither weekend nor holiday is 1, otherwise is 0. + weathersit : - 1: Clear, Few clouds, Partly cloudy, Partly cloudy - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog - temp : Normalized temperature in Celsius. The values are divided to 41 (max) - atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max) - hum: Normalized humidity. The values are divided to 100 (max) - windspeed: Normalized wind speed. The values are divided to 67 (max) - casual: count of casual users - registered: count of registered users - cnt: count of total rental bikes including both casual and registered
=========================================
License
=========================================
Use of this dataset in publications must be cited to the following publication:

[1] Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.

@article{
year={2013},
issn={2192-6352},
journal={Progress in Artificial Intelligence},
doi={10.1007/s13748-013-0040-3},
title={Event labeling combining ensemble detectors and background knowledge},
url={http://dx.doi.org/10.1007/s13748-013-0040-3},
publisher={Springer Berlin Heidelberg},
keywords={Event labeling; Event detection; Ensemble learning; Background knowledge},
author={Fanaee-T, Hadi and Gama, Joao},
pages={1-15}
}

=========================================
Contact
=========================================
For further information about this dataset please contact Hadi Fanaee-T (hadi.fanaee@fe.up.pt)

1

1.  Welcome to the Deep Learning Nanodegree Program
2.  Meet Your Instructors
3.  Program Structure
4.  Community Guidelines
5.  Prerequisites
6.  Getting Set Up

2

3

4

1.  Instructor
2.  Introduction
3.  What is Anaconda?
4.  Installing Anaconda
5.  Managing packages
6.  Managing environments
7.  More environment actions
8.  Best practices
9.  On Python versions at Udacity

5

1.  Introduction
2.  Style Transfer
3.  DeepTraffic
4.  Flappy Bird
5.  Books to Read

6

1.  Instructor
2.  What are Jupyter notebooks?
3.  Installing Jupyter Notebook
4.  Launching the notebook server
5.  Notebook interface
6.  Code cells
7.  Markdown cells
8.  Keyboard shortcuts
9.  Magic keywords
10. Converting notebooks
11. Creating a slideshow
12. Finishing up

7

1.  Introduction
2.  Data Dimensions
3.  Data in NumPy
4.  Element-wise Matrix Operations
5.  Element-wise Operations in NumPy
6.  Matrix Multiplication: Part 1
7.  Matrix Multiplication: Part 2
8.  NumPy Matrix Multiplication
9.  Matrix Transposes
10. Transposes in NumPy
11. NumPy Quiz

2 NEURAL NETWORKS

1

1.  Instructor
2.  Introduction
3.  Classification Problems 1
4.  Classification Problems 2
5.  Linear Boundaries
6.  Higher Dimensions
7.  Perceptrons
8.  Why "Neural Networks"?
9.  Perceptrons as Logical Operators
10. Perceptron Trick
11. Perceptron Algorithm
12. Non-Linear Regions
13. Error Functions
14. Log-loss Error Function
15. Discrete vs Continuous
16. Softmax
17. One-Hot Encoding
18. Maximum Likelihood
19. Maximizing Probabilities
20. Cross-Entropy 1
21. Cross-Entropy 2
22. Multi-Class Cross Entropy
23. Logistic Regression
24. Gradient Descent
25. Logistic Regression Algorithm
26. Pre-Notebook: Gradient Descent
27. Notebook: Gradient Descent
28. Perceptron vs Gradient Descent
29. Continuous Perceptrons
30. Non-linear Data
31. Non-Linear Models
32. Neural Network Architecture
33. Feedforward
34. Backpropagation
35. Pre-Notebook: Analyzing Student Data
36. Notebook: Analyzing Student Data
37. Outro

2

1.  Mean Squared Error Function
2.  Gradient Descent
3.  Gradient Descent: The Math
4.  Gradient Descent: The Code
5.  Implementing Gradient Descent
6.  Multilayer Perceptrons
7.  Backpropagation
8.  Implementing Backpropagation
9.  Further Reading

3

1.  Instructor
2.  Training Optimization
3.  Testing
4.  Overfitting and Underfitting
5.  Early Stopping
6.  Regularization
7.  Regularization 2
8.  Dropout
9.  Local Minima
10. Random Restart
11. Vanishing Gradient
12. Other Activation Functions
13. Batch vs Stochastic Gradient Descent
14. Learning Rate Decay
15. Momentum
16. Error Functions Around the World

4

1.  Introduction to GPU Workspaces
2.  Workspace Playground
3.  GPU Workspace Playground

5

1.  Introducing Andrew Trask
2.  Meet Andrew
3.  Materials
4.  The Notebooks
5.  Framing the Problem
6.  Mini Project 1
7.  Mini Project 1 Solution
8.  Transforming Text into Numbers
9.  Mini Project 2
10. Mini Project 2 Solution
11. Building a Neural Network
12. Mini Project 3
13. Mini Project 3 Solution
14. Better Weight Initialization Strategy
15. Understanding Neural Noise
16. Mini Project 4
17. Understanding Inefficiencies in our Network
18. Mini Project 5
19. Mini Project 5 Solution
20. Further Noise Reduction
21. Mini Project 6
22. Mini Project 6 Solution
23. Analysis: What's Going on in the Weights?
24. Conclusion

6

1.  Introduction to the Project
2.  Project Cheatsheet
3.  Project Workspace
4.  Project: Predicting Bike-Sharing Pa

7

1.  Welcome!
2.  Pre-Notebook
3.  Notebook Workspace
4.  Single layer neural networks
5.  Single layer neural networks solution
6.  Networks Using Matrix Multiplication
7.  Multilayer Networks Solution
8.  Neural Networks in PyTorch
9.  Neural Networks Solution
10. Implementing Softmax Solution
11. Network Architectures in PyTorch
12. Network Architectures Solution
13. Training a Network Solution
14. Classifying Fashion-MNIST
15. Fashion-MNIST Solution
16. Inference and Validation
17. Validation Solution
18. Dropout Solution
19. Saving and Loading Models
20. Loading Image Data
21. Loading Image Data Solution
22. Pre-Notebook with GPU
23. Notebook Workspace w/ GPU
24. Transfer Learning II
25. Transfer Learning Solution
26. Tips, Tricks, and Other Notes

3 CONVOLUTIONAL NEURAL NETWORKS

1

1.  Introducing Alexis
2.  Applications of CNNs
3.  Lesson Outline
4.  MNIST Dataset
5.  How Computers Interpret Images
6.  MLP Structure & Class Scores
7.  Do Your Research
8.  Loss & Optimization
9.  Defining a Network in PyTorch
10. Training the Network
11. Pre-Notebook: MLP Classification, Exercise
12. Notebook: MLP Classification, MNIST
13. One Solution
14. Model Validation
15. Validation Loss
16. Image Classification Steps
17. MLPs vs CNNs
18. Local Connectivity
19. Filters and the Convolutional Layer
20. Filters & Edges
21. Frequency in Images
22. High-pass Filters
23. Quiz: Kernels
24. OpenCV & Creating Custom Filters
25. Notebook: Finding Edges
26. Convolutional Layer
27. Convolutional Layers (Part 2)
28. Stride and Padding
29. Pooling Layers
30. Notebook: Layer Visualization
31. Capsule Networks
32. Increasing Depth
33. CNNs for Image Classification
34. Convolutional Layers in PyTorch
35. Feature Vector
36. Pre-Notebook: CNN Classification
37. Notebook: CNNs for CIFAR Image Classification
38. CIFAR Classification Example
39. CNNs in PyTorch
40. Image Augmentation
41. Augmentation Using Transformations
42. Groundbreaking CNN Architectures
43. Visualizing CNNs (Part 1)
44. Visualizing CNNs (Part 2)
45. Summary of CNNs
46. Introduction to GPU Workspaces
47. Workspace Playground
48. GPU Workspace Playground

2

1.  Mean Squared Error Function
2.  Gradient Descent
3.  Gradient Descent: The Math
4.  Gradient Descent: The Code
5.  Implementing Gradient Descent
6.  Multilayer Perceptrons
7.  Backpropagation
8.  Implementing Backpropagation
9.  Further Reading

3

1.  Instructor
2.  Training Optimization
3.  Testing
4.  Overfitting and Underfitting
5.  Early Stopping
6.  Regularization
7.  Regularization 2
8.  Dropout
9.  Local Minima
10. Random Restart
11. Vanishing Gradient
12. Other Activation Functions
13. Batch vs Stochastic Gradient Descent
14. Learning Rate Decay
15. Momentum
16. Error Functions Around the World

4

1.  Transfer Learning
2.  Useful Layers
3.  Fine-Tuning
4.  VGG Model & Classifier
5.  Pre-Notebook: Transfer Learning
6.  Notebook: Transfer Learning, Flowers
7.  Freezing Weights & Last Layer
8.  Training a Classifier

5

1.  Weight Initialization
2.  Constant Weights
3.  Random Uniform
4.  General Rule
5.  Normal Distribution
6.  Pre-Notebook: Weight Initialization, Normal Distribution
7.  Notebook: Normal & No Initialization
8.  Solution and Default Initialization
9.  Additional Material

6

1. Autoencoders
2. A Linear Autoencoder
3. Pre-Notebook: Linear Autoencoder
4. Notebook: Linear Autoencoder
5. Defining & Training an Autoencoder
6. A Simple Solution
7. Learnable Upsampling
8. Transpose Convolutions
9. Convolutional Autoencoder
10. Pre-Notebook: Convolutional Autoencoder
11. Notebook: Convolutional Autoencoder
12. Convolutional Solution
13. Upsampling & Denoising
14. De-noising
15. Pre-Notebook: De-noising Autoencoder
16. Notebook: De-noising Autoencode

7

1.  Style Transfer
2.  Separating Style & Content
3.  VGG19 & Content Loss
4.  Gram Matrix
5.  Style Loss
6.  Loss Weights
7.  VGG Features
8.  Pre-Notebook: Style Transfer
9.  Notebook: Style Transfer
10. Features & Gram Matrix
11. Gram Matrix Solution
12. Defining the Loss
13. Total Loss & Complete Solution

8

1.  CNN Project: Dog Breed Classifier
2.  Dog Project Workspace
3.  Project: Dog-Breed Classifier

9

1.  Intro
2.  Skin Cancer
3.  Survival Probability of Skin Cancer
4.  Medical Classification
5.  The data
6.  Image Challenges
7.  Quiz: Data Challenges
8.  Solution: Data Challenges
9.  Training the Neural Network
10. Quiz: Random vs Pre-initialized Weights
11. Solution: Random vs Pre-initialized Weight
12. Validating the Training
13. Quiz: Sensitivity and Specificity
14. Solution: Sensitivity and Specificity
15. More on Sensitivity and Specificity
16. Quiz: Diagnosing Cancer
17. Solution: Diagnosing Cancer
18. Refresh on ROC Curves
19. Quiz: ROC Curve
20. Solution: ROC Curve
21. Comparing our Results with Doctors
22. Visualization
23. What is the network looking at?
24. Refresh on Confusion Matrices
25. Confusion Matrix
26. Conclusion
27. Useful Resources
28. Mini Project Introduction
29. Mini Project: Dermatologist AI

10

1. Jobs in Deep Learning
2. Real-World Applications of Deep Lear

11

1.  Prove Your Skills With GitHub
2.  Introduction
3.  GitHub profile important items
4.  Good GitHub repository
5.  Interview with Art - Part 1
6.  Quiz: Identify fixes for example “bad” profile
7.  Quick Fixes #1
8.  Quiz: Quick Fixes #2
9.  Writing READMEs with Walter
10. Interview with Art - Part 2
11. Commit messages best practices
12. Quiz: Reflect on your commit messages
13. Participating in open source projects
14. Interview with Art - Part 3
15. Participating in open source projects 2
16. Quiz: Starring interesting repositories
17. Next Steps
18. Project: Optimize Your GitHub Pro

4 RECURRENT NEURAL NETWORKS

1

1.  RNN Examples
2.  RNN Introduction
3.  RNN History
4.  RNN Applications
5.  Feedforward Neural Network-Reminder
6.  The Feedforward Process
7.  Feedforward Quiz
8.  Backpropagation- Theory
9.  Backpropagation - Example (part a)
10. Backpropagation- Example (part b)
11. Backpropagation Quiz
12. RNN (part a)
13. RNN (part b)
14. RNN- Unfolded Model
15. Unfolded Model Quiz
16. RNN- Example
17. Backpropagation Through Time (part a)
18. Backpropagation Through Time (part b)
19. Backpropagation Through Time (part c)
20. BPTT Quiz 1
21. BPTT Quiz 2
22. BPTT Quiz 3
23. Some more math
24. RNN Summary
25. From RNN to LSTM
26. Wrap Up

2

1.  Intro to LSTM
2.  RNN vs LSTM
3.  Basics of LSTM
4.  Architecture of LSTM
5.  The Learn Gate
6.  The Forget Gate
7.  The Remember Gate
8.  The Use Gate
9.  Putting it All Together
10. Quiz
11. Other architectures

3

1.  Implementing RNNs
2.  Time-Series Prediction
3.  Training & Memory
4.  Character-wise RNNs
5.  Sequence Batching
6.  Pre-Notebook: Character-Level RNN
7.  Notebook: Character-Level RNN
8.  Implementing a Char-RNN
9.  Batching Data, Solution
10. Defining the Model
11. Char-RNN, Solution
12. Making Predictions

4

1.  Introducing Jay
2.  Introduction
3.  Learning Rate
4.  Quiz: Learning Rate
5.  Minibatch Size
6.  Number of Training Iterations / Epochs
7.  Number of Hidden Units / Layers
8.  RNN Hyperparameters
9.  Quiz: RNN Hyperparameters
10. Sources & References

5

1.  Word Embeddings
2.  Embedding Weight Matrix/Lookup Table
3.  Word2Vec Notebook
4.  Pre-Notebook: Word2Vec, SkipGram
5.  Notebook: Word2Vec, SkipGram
6.  Data & Subsampling
7.  Subsampling Solution
8.  Context Word Targets
9.  Batching Data, Solution
10. Word2Vec Model
11. Model & Validations
12. Negative Sampling
13. Pre-Notebook: Negative Sampling
14. Notebook: Negative Sampling
15. SkipGramNeg, Model Definition
16. Complete Model & Custom Loss

6

1.  Sentiment RNN, Introduction
2.  Pre-Notebook: Sentiment RNN
3.  Notebook: Sentiment RNN
4.  Data Pre-Processing
5.  Encoding Words, Solution
6.  Getting Rid of Zero-Length
7.  Cleaning & Padding Data
8.  Padded Features, Solution
9.  TensorDataset & Batching Data
10. Defining the Model
11. Complete Sentiment RNN
12. Training the Model
13. Testing
14. Inference, Solution

7

1. Introduction
2. GPU Workspaces: Best Practices
3. Project Workspace: TV Script Generation
4. Project: Generate TV Scripts

8

1.  Introduction to Attention
2.  Encoders and Decoders
3.  Sequence to Sequence Recap
4.  Encoding -- Attention Overview
5.  Decoding -- Attention Overview
6.  Quiz: Attention Overview
7.  Attention Encoder
8.  Attention Decoder
9.  Quiz: Attention Encoder & Decoder
10. Bahdanau and Luong Attention
11. Multiplicative Attention
12. Additive Attention
13. Quiz: Additive and Multiplicative Attention
14. Computer Vision Applications
15. Other Attention Methods
16. The Transformer and Self-Attention
17. Notebook: Attention Basics
18. [SOLUTION]: Attention Basics
19. Outro

5 GENERATIVE ADVERSARIAL NETWORKS

1

1.  Introducing Ian GoodFellow
2.  Applications of GANs
3.  How GANs work
4.  Games and Equilibria
5.  Tips for Training GANs
6.  Generating Fake Images
7.  MNIST GAN
8.  GAN Notebook & Data
9.  Pre-Notebook: MNIST GAN
10. Notebook: MNIST GAN
11. The Complete Model
12. Generator & Discriminator
13. Hyperparameters
14. Fake and Real Losses
15. Optimization Strategy, Solution
16. Training Two Networks
17. Training Solution

2

1.  Deep Convolutional GANs
2.  DCGAN, Discriminator
3.  DCGAN Generator
4.  What is Batch Normalization?
5.  Pre-Notebook: Batch Norm
6.  Notebook: Batch Norm
7.  Benefits of Batch Normalization
8.  DCGAN Notebook & Data
9.  Pre-Notebook: DCGAN, SVHN
10. Notebook: DCGAN, SVHN
11. Scaling, Solution
12. Discriminator
13. Discriminator, Solution
14. Generator
15. Generator, Solution
16. Optimization Strategy
17. Optimization Solution & Samples
18. Other Applications of GANs

3

1.  Introducing Jun-Yan Zhu
2.  Image to Image Translation
3.  Designing Loss Functions
4.  GANs, a Recap
5.  Pix2Pix Generator
6.  Pix2Pix Discriminator
7.  CycleGANs & Unpaired Data
8.  Cycle Consistency Loss
9.  Why Does This Work?
10. Beyond CycleGANs

4

1.  CycleGAN Notebook & Data
2.  Pre-Notebook: CycleGAN
3.  Notebook: CycleGAN
4.  DC Discriminator
5.  DC Discriminator, Solution
6.  Generator & Residual Blocks
7.  CycleGAN Generator
8.  Blocks & Generator, Solution
9.  Adversarial & Cycle Consistency Losses
10. Loss & Optimization, Solution
11. Training Exercise
12. Training Solution & Generated Sam

5

1. Project Introduction
2. Project Instructions
3. Face Generation Workspace
4. Project: Generate Faces

6

1.  Get Opportunities with LinkedIn
2.  Use Your Story to Stand Out
3.  Why Use an Elevator Pitch
4.  Create Your Elevator Pitch
5.  Use Your Elevator Pitch on LinkedIn
6.  Create Your Profile With SEO In Mind
7.  Profile Essentials
8.  Work Experiences & Accomplishments
9.  Build and Strengthen Your Network
10. Reaching Out on LinkedIn
11. Boost Your Visibility
12. Up Next
13. Project: Improve Your LinkedIn Profi

6 DEPLOYING A MODEL

1

1.  Welcome!
2.  What's Ahead?
3.  Problem Introduction
4.  Machine Learning Workflow
5.  Quiz: Machine Learning Workflow
6.  What is Cloud Computing & Why Would We Use It?
7.  Why Cloud Computing?
8.  Machine Learning Applications
9.  Machine Learning Applications
10. Paths to Deployment
11. Quiz: Paths to Deployment
12. Production Environments
13. Production Environments
14. Endpoints & REST APIs
15. Endpoints & REST APIs
16. Containers
17. Quiz: Containers
18. Containers - Straight From the Experts
19. Characteristics of Modeling & Deployment
20. Characteristics of Modeling & Deployment
21. Comparing Cloud Providers
22. Comparing Cloud Providers
23. Closing Statements
24. Summary
25. [Optional] Cloud Computing Defined
26. [Optional] Cloud Computing Explai

2

1.  Introduction to Amazon SageMaker
2.  AWS Setup Instructions
3.  Get Access to GPU Instances
4.  Setting up a Notebook Instance
5.  Cloning the Deployment Notebooks
6.  Is Everything Set Up?
7.  Boston Housing Example - Getting the Data Ready
8.  Boston Housing Example - Training the Model
9.  Boston Housing Example - Testing the Model
10. Mini-Project: Building Your First Model
11. Mini-Project: Solution
12. Boston Housing In-Depth - Data Preparation
13. Boston Housing In-Depth - Creating a Training Job
14. Boston Housing In-Depth - Building a Model
15. Boston Housing In-Depth - Creating a Batch Transform Job
16. Summary

3

1.  Deploying a Model in SageMaker
2.  Boston Housing Example - Deploying the Model
3.  Boston Housing In-Depth - Deploying the Model
4.  Deploying and Using a Sentiment Analysis Model
5.  Text Processing, Bag of Words
6.  Building and Deploying the Model
7.  How to Use a Deployed Model
8.  Creating and Using an Endpoint
9.  Building a Lambda Function
10. Building an API
11. Using the Final Web Application
12. Summary

4

1.  Hyperparameter Tuning
2.  Introduction to Hyperparameter Tuning
3.  Boston Housing Example - Tuning the Model
4.  Mini-Project: Tuning the Sentiment Analysis Model
5.  Mini-Project: Solution - Tuning the Model
6.  Mini-Project: Solution - Fixing the Error and Testing
7.  Boston Housing In-Depth - Creating a Tuning Job
8.  Boston Housing In-Depth - Monitoring the Tuning Job
9.  Boston Housing In-Depth - Building and Testing the Model
10. Summary

5

1. Updating a Model
2. Building a Sentiment Analysis Model (XGBoost)
3. Building a Sentiment Analysis Model (Linear Learner)
4. Combining the Models
5. Mini-Project: Updating a Sentiment Analysis Model
6. Loading and Testing the New Data
7. Exploring the New Data
8. Building a New Model
9. SageMaker Retrospective
10. Cleaning Up Your AWS Account
11. SageMaker Tips and Tricks

6

1.  Deployment Project
2.  Setting up a Notebook Instance
3.  Get Access to GPU Instances
4.  Project: Deploying a Sentiment Analy

EXTRACURRICULAR

1 ADDITIONAL LESSONS

1

1.  Intro
2.  Confusion Matrix
3.  Confusion Matrix 2
4.  Accuracy
5.  Accuracy 2
6.  When accuracy won't work
7.  False Negatives and Positives
8.  Precision and Recall
9.  Precision
10. Recall
11. ROC Curve

2

1.  Intro
2.  Quiz: Housing Prices
3.  Solution: Housing Prices
4.  Fitting a Line Through Data
5.  Moving a Line
6.  Absolute Trick
7.  Square Trick
8.  Gradient Descent
9.  Mean Absolute Error
10. Mean Squared Error
11. Minimizing Error Functions
12. Mean vs Total Error
13. Mini-batch Gradient Descent
14. Absolute Error vs Squared Error
15. Linear Regression in scikit-learn
16. Higher Dimensions
17. Multiple Linear Regression
18. Closed Form Solution
19. (Optional) Closed form Solution Math
20. Linear Regression Warnings
21. Polynomial Regression
22. Regularization
23. Neural Network Regression
24. Neural Networks Playground
25. Outro

3

1. Welcome to MiniFlow
2. Graphs
3. MiniFlow Architecture
4. Forward Propagation
5. Forward Propagation Solution
6. Learning and Loss
7. Linear Transform
8. Sigmoid Function
9. Cost
10. Cost Solution
11. Gradient Descent
12. Backpropagation
13. Stochastic Gradient Descent
14. SGD Solution
15. Outro

2 TENSORFLOW KERAS FRAMEWORKS

1

1.  Intro
2.  Keras
3.  Pre-Lab: Student Admissions in Keras
4.  Lab: Student Admissions in Keras
5.  Optimizers in Keras
6.  Mini Project Intro
7.  Pre-Lab: IMDB Data in Keras
8.  Lab: IMDB Data in Keras

2

1. Convolutional Layers in Keras
2. Quiz: Dimensionality
3. CNNs in Keras: Practical Example
4. Mini Project: CNNs in Keras
5. Image Augmentation in Keras
6. Mini Project: Image Augmentation in Keras
7. Transfer Learning
8. Transfer Learning in Keras

3

1. Intro
2. Installing TensorFlow
3. Hello, Tensor World!
4. Quiz: TensorFlow Input
5. Quiz: TensorFlow Math
6. Quiz: TensorFlow Linear Function
7. Quiz: TensorFlow Softmax
8. Quiz: TensorFlow Cross Entropy
9. Quiz: Mini-batch
10. Epochs
11. Pre-Lab: NotMNIST in TensorFlow
12. Lab: NotMNIST in TensorFlow
13. Two-layer Neural Network
14. Quiz: TensorFlow ReLUs
15. Deep Neural Network in TensorFlow
16. Save and Restore TensorFlow Models
17. Finetuning
18. Quiz: TensorFlow Dropout
19. Outro
