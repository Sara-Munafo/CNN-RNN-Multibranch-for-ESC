# CNN-RNN-Multibranch-for-ESC

This repository contains the model I developed for the Machine Learning for Human Data course of the Physics of Data Master's Degree at the University of Padua.    
The project, inspired mainly by the pioneering work of Piczak et. al. (https://www.karolpiczak.com/papers/Piczak2015-ESC-ConvNet.pdf), who were among the firsts to approach the Environmental Sound Classification task by using a CNN based architecture, ConvNet.    
In my project, I worked with the ESC-50 dataset developed and used by the authors (https://www.karolpiczak.com/papers/Piczak2015-ESC-Dataset.pdf), consisting of 50 classes of various environmental sounds.    
![Senza titolo|100](https://github.com/user-attachments/assets/c281f008-b540-4cd9-86eb-00e0a865dd03)

My architecture consists in a Multi-branch architecture where a CNN and a RNN work in parallel to process different features extracted from the raw audio data (Mel-spectrograms and MFCCs respectively); the outputs are then concatenated before the Fully Connected Layers and Output (Softmax) Layer.    
Different variants of this architectures were trained and compared, obtaining a final architecture that managed to improve the original's ConvNet performance on the test dataset.

In this repository you will find:
- The Python modules implemented for the different stages of development
  - *data_helper.py*: contains the Clips class and all the functions used to handle data in the first stages of the project (functions to download the dataset, load it, organize it into folders, split it into train/test and data augmentation);
  - *preprocess.py*: contains all the functions used for preprocessing, such as standardization, segmentation of the features and organization into folds;
  - *train_test_model.py*: contains the functions used to train the models, as well as the functions implemented to aggregate predictions and compute per-class accuracy in the test stage;
  - *plot_utils.py*: contains all the functions related to visualization, from the first stages to the final models comparison.
- The main Jupyter notebook, *CRNN-Mb.ipynb* where all training and module selection process is explained.
- The .h5 file with the best overall performing variant of the model (1000 units per FC layer, 32 bi-GRU units) *cnn2d_rnn_multibranch_32_1000_T.h5*
- The *DEMO.ipynb* notebook, where the final model accuracy is seen in detail for a selection of classes from the test dataset.
- The *requirements.txt* text files with all the requirements needed to run the notebooks.
