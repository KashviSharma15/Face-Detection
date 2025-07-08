# Face-Detection
<img src="https://github.com/user-attachments/assets/080706db-9d29-4304-9ad9-4ac29e5ce29d" width="300"/>

<br>
The goal of this project was to build a robust and accurate deep learning model that can classify input images as face or not using a dataset sourced from Kaggle. In this we have dataset which had several face and non-face images(cat ,dog, monkey, horse , flower etc) and the model is then trained to detect whether the image is a human face or not. A Convolutional Neural Network (CNN) was employed as it is particularly effective for spatial feature extraction and classification tasks involving images.<br>
<b>Dataset :</b>
https://www.kaggle.com/datasets/sagarkarar/nonface-and-face-dataset <br>
**Apart from this I have added few more images from other dataset and my personal folders as well , to add variety and balance the number of face and non-face images.

<br>
<b>Library :</b> <br>
os 
<br>
numpy
<br>
pandas
<br>
matplotlib
<br>
Pillow (PIL)
<br>
tensorflow
<br>
scikit-learn (sklearn)
<br>
warnings
<br>
<b>Algorithm Used</b> <br>
Convolutional Neural Network (CNN) :
The central algorithm used in this face detection project is the Convolutional Neural Network (CNN), a specialized type of deep neural network particularly effective for image recognition and classification. CNNs are designed to automatically and adaptively learn spatial hierarchies of features through a series of convolutional layers, pooling layers, and fully connected layers. In this project, CNNs played a crucial role in classifying facial images from the Kaggle dataset into two classes (such as face vs. nonface). The network structure included multiple core components, each serving a distinct purpose within the CNN pipeline.
<br>
<b>Evaluation Metrics Used :</b>
<br>
Accuracy 
 <br>
Classification Report
<br>
Confusion Matrix 


