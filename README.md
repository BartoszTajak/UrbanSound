## Urban   Sound
Program to recognize the type of sound 

Samples download from website : 

https://www.kaggle.com/chrisfilo/urbansound8k

And collected by me from www.youtube.com 

There are more than 14k samples divided into 16 classes. Each sample has 4sec. duration
## Features extracted from samples

1. Zero crossing rate
2. Energy Value
3. Root mean square energy
4. Spectral centroid
5. MFCC
6. Delta MFCC
7. Chroma STFT
8. RollOff
9. Spectral flux
10. Spectral Bandwidth
## 


Below some screenshots show a visualization of features

![Przechwytywanie4](https://user-images.githubusercontent.com/67312266/152697595-b9bf4e27-2b1a-4e77-bd1c-44bd6ca749ea.PNG)
![Przechwytywanie](https://user-images.githubusercontent.com/67312266/152697596-8cf6c5df-26da-4a7a-99ef-a584d3418672.PNG)
![Przechwytywanie2](https://user-images.githubusercontent.com/67312266/152697597-cfb8b0f5-4400-4f5f-bfe7-b3a79d5b0136.PNG)
![Przechwytywanie3](https://user-images.githubusercontent.com/67312266/152697598-d67735af-df0e-4541-bda4-82c6fa4dc0a3.PNG)
## Machine Learning Models

For classification were used models : 

1. Decision Tree Classifier
2. Xgboost Classification
3. Random Forest Classifier
4. SVC Classification
5. Neutral Network


## Measure Classification Performance
Accuracy is messue by the following methods: 
1. Matthews correlation coefficient (MCC)
2. Cohen kappa score
3. Classification report
4. Confusion matrix
5. ROC AUC


## Libraries 
1. Librosa - 0.8.1
2. psycopg2 - 2.9.3
3. tensorflow - 2.7
4. sklearn - 1.0.2
5. numpy - 1.20.3
6. pandas - 1.3.4
7. keras - 2.7