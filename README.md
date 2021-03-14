# Depression Detection on Twitter using hybrid CNN-LSTM Model
Social media provides an unprecedented opportunity to transform early depression
intervention services, particularly in young adults. Because depression is an illness that
so often requires the self-reporting of symptoms, social media posts provide a rich source
of data and information that can be used to train an eficient deep learning model. Social
networks have been developed as a great point for its users to communicate with their
interested friends and share their opinions, photos, and videos reflecting their moods,
feelings and sentiments. This creates an opportunity to analyze social network data for
user's feelings and sentiments to investigate their moods and attitudes when they are
communicating via these online tools.
We aim to perform depression analysis on Twitter by analyzing its linguistic markers
which makes it possible to create a deep learning model that can give an individual insight
into his or her mental health far earlier than traditional approaches. On this conquest a
CNN-LSTM model is proposed to detect depressive users using their tweets. Two other
models are also used, a simple RNN and LSTM. The proposed model outperforms them
by having a validation accuracy of 96.28% and validation loss of 0.1071. All the 3 models
were trained on a single customized dataset, half of which was from sentiment140, and
the other half was extracted from Twitter using the Twitter API. Moreover, this work
shows how adding a Convolutional layer to LSTM improves the accuracy.



## Dataset
Two types of tweets were utilized in order to build the model: random tweets which do not necessarily indicate depression [Sentiment140 data-set on kaggle](https://www.kaggle.com/kazanova/sentiment140) and [depressed tweets](https://github.com/aapoorv-tf/depdetection/tree/master/datasets) which were extracted using the twitter API. Since there are no publicly available twitter data-set that indicates depression, the tweets were extracted according to the linguistic markers indicative of depression such as ”Hopeless”, ”Lonely”, ”Suicide”, ”Antidepressants”, ”Depressed”, etc. 

## Visualisation
Once the Tweets were tokenized, lemmatized and cleaned, it was easy to see the difference between the two datasets
by creating a word cloud with the cleaned Tweets. With only an abbreviated Twitter
scraping, the differences between the two datasets were clear: \
<img alt="Depressed Tweets" src="https://github.com/aapoorv-tf/depdetection/blob/master/img/depressedWordcloud.png" width='400'>
<img alt="Random Tweets" src="https://github.com/aapoorv-tf/depdetection/blob/master/img/normalWordCloud.png" width='400'>

## Building the Models
Once the data was fully processed, each word had to be converted into its corresponding
index of the word embedding, as the model can take only numbers as input. Words that
didn't occur in the word embedding were given the index of "UNK".
To build the model we used the keras API. [Keras](https://keras.io/) is an open-source software library
that provides a Python interface for artificial neural networks. It acts as an interface
for the TensorFlow library. The list is passed to the embedding layer which is a part
of the model. This layer will look up the index in the gloVe embeddings and give its
corresponding 300 dimensional vector. Next we define 3 sequencial models, first one is a
simple RNN model, second one is LSTM, and the last one is CNN-LSTM (composed of
a 1D convolutional layer, max pooling layer and LSTM).

### Simple RNN
<img alt="Simple RNN model" src="https://github.com/aapoorv-tf/depdetection/blob/master/img/SimpleRNNmodel.png" width='200'>

**Table 1:** Simple RNN summmary

|  Layer | Outer Shape | Parameters |
|:----------------:| :-------:| :------:|
| Embedding | (None, 49, 300) | 120000300 |
| Simple RNN  | (None, 64) | 23360 |
| Dropout | (None, 64) | 0 |
| Dense | (None, 1) | 65 |

Total params: 120,023,725 \
Trainable params: 23,425 \
Non-trainable params: 120,000,300 

### LSTM
<img alt="LSTM model" src="https://github.com/aapoorv-tf/depdetection/blob/master/img/LSTMmodel.png" width='200'>

**Table 2:** LSTM summmary

|  Layer | Outer Shape | Parameters |
|:----------------:| :-------:| :------:|
| Embedding | (None, 49, 300) | 120000300 |
| LSTM  | (None, 16) | 20288 |
| Dropout | (None, 16) | 0 |
| Dense | (None, 1) | 17 |

Total params: 120,020,605 \
Trainable params: 20,305 \
Non-trainable params: 120,000,300 

### CNN-LSTM
<img alt="CNN-LSTM Model" src="https://github.com/aapoorv-tf/depdetection/blob/master/img/model.png" width='200'>

**Table 3:** CNN-LSTM summmary

|  Layer | Outer Shape | Parameters |
|:----------------:| :-------:| :------:|
| Embedding | (None, 49, 300) | 120000300 |
| Conv1D | (None, 49, 32) | 28832 |
| MaxPooling1D | (None, 24, 32) | 0 |
| Dropout | (None, 24, 32) | 0 |
| LSTM | (None, 300) | 399600 |
| Dropout | (None, 300) | 0 |
| Dense | (None, 1) | 301 |

Total params: 120,429,033 \
Trainable params: 428,733 \
Non-trainable params: 120,000,300 

The data was split into 60% train, 20% validation and 20% test data. We used
the nadam optimizer. Adam optimizer is a stochastic gradient descent method that
is based on adaptive estimation of first-order and second-order moments. Also, it is
computationally eficient, has little memory requirement, invariant to diagonal rescaling
of gradients, and is well suited for problems that are large in terms of data/parameters. Much like Adam is essentially RMSprop with momentum, Nadam is Adam with
Nesterov momentum.

## Results

Once the models were compiled, there are now ready to be trained using the fit() function of the Sequential class of keras API. Since too many epochs can lead to overfitting of the training dataset, whereas too few may result in an underfit model, we give the argument of early stopping to the ”callbacks”, which provides a way to execute code and interact with the training model process automatically, Early stopping is a method that allows you to specify an arbitrary large number of training epochs and stop training once the model performance stops improving on a holdout validation dataset.
Models were trained for 20 epochs before which the training was automatically called
off since the validation accuracy started to drop from there on. The training and validation accuracy/loss of all the models are compared in the following figures. For the simple RNN model the validation accuracy started from 94.01% on the first epoch and reached 95.90% on the 17th epoch. Validation loss was 0.1765 on first epoch which was reduced to 0.1176 on the last epoch. For LSTM model, the validation accuracy increased from 94.25% to 96.22% in 20 epochs and validation loss reduced from 0.2106 to 0.1077. The final model, CNN-LSTM outperformed the previous models. It managed to reach 96.28% validation accuracy in the 11th epoch with validation loss of 0.1071. Moreover, the overfitting was quite low with training accuracy of 96.56%.

<img alt="Simple RNN Model Acc" src="https://github.com/aapoorv-tf/depdetection/blob/master/img/RNNaccuracy.png" width='250'>
<img alt="LSTM Model Acc" src="https://github.com/aapoorv-tf/depdetection/blob/master/img/LSTMaccuracy.png" width='250'>
<img alt="CNN-LSTM Model acc" src="https://github.com/aapoorv-tf/depdetection/blob/master/img/accuracy.png" width='250'>
<img alt="Simple RNN Model loss" src="https://github.com/aapoorv-tf/depdetection/blob/master/img/RNNloss.png" width='250'>
<img alt="LSTM Model loss" src="https://github.com/aapoorv-tf/depdetection/blob/master/img/LSTMloss.png" width='250'>
<img alt="CNN-LSTM Model loss" src="https://github.com/aapoorv-tf/depdetection/blob/master/img/loss.png" width='250'>


Since accuracy is not always the metric to determine how good the model is. Therefore, 3 other metrics, precision, recall and F1-score are also calculated.

**Table 4:** Simple RNN Classification Report
|      | Precision | Recall | F1-score | support |
|:----:|:--------:| :--------:| :-----:| :-------:|
| Class 0 | 0.97   | 0.95    | 0.96  | 29314    |
| Class 1 | 0.95  | 0.97    | 0.96  | 29877   |
| Accuracy |   |    | 0.96  | 59191   |
| Macro Avg. | 0.96   | 0.96    | 0.96  | 59191   |
| Weighted Avg. | 0.96  | 0.96    | 0.96  | 59191   |

**Table 5:** LSTM summmary Classification Report
|      | Precision | Recall | F1-score | support |
|:----:|:--------:| :--------:| :-----:| :-------:|
| Class 0 | 0.98   | 0.95    | 0.96  | 29314    |
| Class 1 | 0.95  | 0.98    | 0.97  | 29877   |
| Accuracy |   |    | 0.96  | 59191   |
| Macro Avg. | 0.96   | 0.96    | 0.96  | 59191   |
| Weighted Avg. | 0.96  | 0.96    | 0.96  | 59191   |

**Table 6:** CNN-LSTM summmary Classification Report
|      | Precision | Recall | F1-score | support |
|:----:|:--------:| :--------:| :-----:| :-------:|
| Class 0 | 0.99   | 0.94    | 0.96  | 29314    |
| Class 1 | 0.94  | 0.99    | 0.96  | 29877   |
| Accuracy |   |    | 0.96  | 59191   |
| Macro Avg. | 0.96   | 0.96    | 0.96  | 59191   |
| Weighted Avg. | 0.96  | 0.96    | 0.96  | 59191   |

It clearly shows that the best result is achieved by model 3 (CNN-LSTM) which
is trained and validated on 150,000 depressive tweets and 150,000 random tweets with
accuracy and F1-score of 96.28% and 96% respectively. It performs better than other the
models.

| Models |  Val. Acc    | Precision | Recall | F1-score | Loss |
|:---:|:----:|:--------:| :--------:| :-----:| :-------:|
| Simple RNN | 95.98% | 0.96   | 0.96    | 0.96  | 0.1176    |
| LSTM | 96.21% | 0.96 | 0.96  | 0.96  | 0.1077  |
| CNN-LSTM | 96.28% |  0.96 |   0.96 | 0.96  | 0.1071   |



## References
    1. Gratch, Artstein, Lucas, Stratou, Scherer, Nazarian, Wood, Boberg, DeVault, Marsella, Traum. The Distress Analysis Interview Corpus of human and computer interviews. InLREC 2014 May (pp. 3123-3128).
    2. Girard, Cohn. Automated Depression Analysis. Curr Opin Psychol. 2015 August; 4: 75–79.
    3. Ma, Yang, Chen, Huang, and Wang. DepAudioNet: An Efficient Deep Model for Audio based Depression Classification. ACM International Conference on Multimedia (ACM-MM) Workshop: Audio/Visual Emotion Challenge (AVEC), 2016.
