# Emotion Recognition on Masked Faces

Face detection algorithms have become ubiquitous and the need for human emotion detection is only on the rise. Studies have shown that computer vision and various machine learning techniques can be applied to the identification and classification of such emotions. However, as it is becoming increasingly common for individuals to wear face masks in their day-to-day life. Our goal is to use a neural network to implement a deep learning network that identifies emotions from pictures of a masked individual’s face.

## Setup

Use the requirements.txt to install the appropiate libraries for preprocessing datasets. You must have PyTorch installed in order to train or use a model, so please follow the proper installation for you machine from [pytorch.org](pytorch.org). The project was run using Python 3.8 and PyTorch 1.7.1.
## Data Processing

Main dataset: [Kaggle Facial Expression Recognition Challenge dataset](https://www.kaggle.com/debanga/facial-expression-recognition-challenge)

Pre-processing python script: [data_processing/data_kaggle.py](https://github.com/rebyu/masked-emotions/blob/master/data_processing/data_kaggle.py)

The pre-processing script first loads in the train/test/validation splits of data (image name and emotion label). Then the script uses the dlib facial feature detector to landmark points on the faces in the images and generates a mask onto the images. These images (saved as jpgs) and associated emotion labels (saved in a numpy array) are stored in a generated images folder.

## Models

Model training python script: [train_model.py](https://github.com/rebyu/masked-emotions/blob/master/train_model.py)

Use the `--help` flag to see the possible parameters you can pass to train a model. An example usage of the command would be:

```
  python .\model_train.py --model alexnet --model_dir ./test --batch_size 24 --learning_rate 0.0025 --num_epochs 10 --data_loc ..\..\images\ --verbose --pretrained --data_aug --save_predictions
```

### Our Models:

- AlexNet, lr = 0.005, 150 epochs
- VGG-11, lr = 0.005, 150 epochs
- VGG-19, lr = 0.01, 200 epochs
- Resnet-18, lr = 0.005, 150 epochs
- DenseNet-121, lr = 0.001, 150 epochs
- DenseNet-121, lr = 0.0025, 150 epochs
- DenseNet-121, lr = 0.01, 200 epochs

These models are saved in our google drive in a zip.

## Data Analysis

Data analysis python notebook: [analyze_model.ipynb](https://github.com/rebyu/masked-emotions/blob/master/analyze_model.ipynb)

The python notebook takes the true labels from the dataset and the predicted labels of our models to create:

- Confusion matrix of the predicted labels
- Bar plots of the distribution of true and predicted labels
- Overall accuracy score of the predicted labels

The predicted labels and resulting data analysis graphs for each model can be found in the google drive, with their respectively named folders.

## Simple Testing

Notebook for generating predictions: [run_model.ipynb](https://github.com/rebyu/masked-emotions/blob/master/run_model.ipynb)

There are fields to allow you to choose a model to run and what image to test on, which are indicated in the notebook. Simply run the entire notebook to generate the models prediction on the example.

## Credits

Collaborators: Rebecca Yu, Silu Men, Eugene Asare, Jennifer Lin

[Google Drive Link](https://drive.google.com/drive/folders/1gMW66r3nVHQshfGd1qnWm159tcPMVSmy?usp=sharing)
