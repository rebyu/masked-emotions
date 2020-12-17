# Emotion Recognition on Masked Faces

Face detection algorithms have become ubiquitous and the need for human emotion detection is only on the rise. Studies have shown that computer vision and various machine learning techniques can be applied to the identification and classification of such emotions. However, as it is becoming increasingly common for individuals to wear face masks in their day-to-day life. Our goal is to use a neural network to implement a deep learning network that identifies emotions from pictures of a masked individual’s face.

## Data Processing

Main dataset: [Kaggle Facial Expression Recognition Challenge dataset](https://www.kaggle.com/debanga/facial-expression-recognition-challenge)

Pre-processing python script: data_processing/data_zenify/data_kaggle.py

The pre-processing script first loads in the train/test/validation splits of data (image name and emotion label). Then the script uses the dlib facial feature detector to landmark points on the faces in the images and generates a mask onto the images. These images (saved as jpgs) and associated emotion labels (saved in a numpy array) are stored in a generated images folder.

## Model

Model training python script: model/model_train.py

## Data Analysis

Data analysis python notebook: data_analysis/analysis.ipynb

The python notebook takes the true labels from the dataset and the predicted labels of our models to create:
- Confusion matrix of the predicted labels
- Bar plots of the distribution of true and predicted labels
- Overall accuracy score of the predicted labels
The predicted labels and resulting data analysis graphs for each model can be found in their respectively named folders.

## Credits
Collaborators: Rebecca Yu, Silu Men, Eugene Asare, Jennifer Lin

[Google Drive Link](https://drive.google.com/drive/folders/1CGh-vtHR73mHsYgof3eQf965zkD4R2sx)
