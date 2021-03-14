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
scraping, the differences between the two datasets were clear:
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
<img alt="Random Tweets" src="https://github.com/aapoorv-tf/depdetection/blob/master/img/SimpleRNNmodel.png" width='400'>

### Results

**Table 1:** Test set predictions on 4-second spectrograms

|  Confusion Matrix | Actual: Yes | Actual: No |
|:----------------:| :-------:| :------:|
| **Predicted: Yes**  | 174 (TP) | 106 (FP) |
| **Predicted: No**   | 144 (FN) | 136 (TN) |

| F1 score | precision | recall | accuracy |
|:--------:| :--------:| :-----:| :-------:|
| 0.582    | 0.621     | 0.547  | 0.555    |


<img alt="ROC curve" src="images/roc_curve.png" width='550'>

<sub><b>Figure 6: </b> ROC curve of the CNN model. </sub>


**Table 2:** Test set predictions using majority vote

|  Confusion Matrix | Actual: Yes | Actual: No |
|:----------------:| :-------:| :------:|
| **Predicted: Yes**  | 4 (TP) | 2 (FP) |
| **Predicted: No**   | 3 (FN) | 5 (TN) |

| F1 score | precision | recall | accuracy |
|:--------:| :--------:| :-----:| :-------:|
| 0.615    | 0.667     | 0.571  | 0.643    |



## References
    1. Gratch, Artstein, Lucas, Stratou, Scherer, Nazarian, Wood, Boberg, DeVault, Marsella, Traum. The Distress Analysis Interview Corpus of human and computer interviews. InLREC 2014 May (pp. 3123-3128).
    2. Girard, Cohn. Automated Depression Analysis. Curr Opin Psychol. 2015 August; 4: 75–79.
    3. Ma, Yang, Chen, Huang, and Wang. DepAudioNet: An Efficient Deep Model for Audio based Depression Classification. ACM International Conference on Multimedia (ACM-MM) Workshop: Audio/Visual Emotion Challenge (AVEC), 2016.
    4. Giannakopoulos, Aggelos. Introduction to audio analysis: a MATLAB approach. Oxford: Academic Press, 2014.
    5. Piczak. Environmental Sound Classification with Convolutional Neural Networks. Institute of Electronic System, Warsaw University of Technology, 2015.

## Code References
    1. http://yerevann.github.io/2015/10/11/spoken-language-identification-with-deep-convolutional-networks
    2. http://www.frank-zalkow.de/en/code-snippets/create-audio-spectrograms-with-python.html
