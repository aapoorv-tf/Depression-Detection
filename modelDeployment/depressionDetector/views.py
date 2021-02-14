from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import twitter_id
from decouple import config
import tweepy as tw
import numpy as np
import nltk
import re, string
import tensorflow as tf
from nltk.corpus import stopwords
nltk.download('stopwords')
STOP_WORDS = stopwords.words('english')
from nltk.tokenize import TweetTokenizer
tk = TweetTokenizer(reduce_len=True)


consumer_key = config('API_KEY')
consumer_secret = config('API_KEY_SECRET')
access_key = config('ACCESS_TOKEN')
access_secret = config('ACCESS_TOKEN_SECRET')

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tw.API(auth, wait_on_rate_limit=True)

def cleaned(token):
    if token == 'u':
        return 'you'
    if token == 'r':
        return 'are'
    if token == 'some1':
        return 'someone'
    if token == 'yrs':
        return 'years'
    if token == 'hrs':
        return 'hours'
    if token == 'mins':
        return 'minutes'
    if token == 'secs':
        return 'seconds'
    if token == 'pls' or token == 'plz':
        return 'please'
    if token == '2morow' or token == '2moro':
        return 'tomorrow'
    if token == '2day':
        return 'today'
    if token == '4got' or token == '4gotten':
        return 'forget'
    if token in ['hahah', 'hahaha', 'hahahaha']:
        return 'haha'
    if token == "mother's":
        return "mother"
    if token == "mom's":
        return "mom"
    if token == "dad's":
        return "dad"
    if token == 'bday' or token == 'b-day':
        return 'birthday'
    if token in ["i'm", "don't", "can't", "couldn't", "aren't", "wouldn't", "isn't", "didn't", "hadn't", "doesn't",
                 "won't", "haven't", "wasn't", "hasn't", "shouldn't", "ain't", "they've"]:
        return token.replace("'", "")
    if token in ['lmao', 'lolz', 'rofl']:
        return 'lol'
    if token == '<3':
        return 'love'
    if token == 'thanx' or token == 'thnx' or token == "thnks":
        return 'thanks'
    if token == 'goood':
        return 'good'
    if token in ['amp', 'quot', 'lt', 'gt', '½25', '..', '. .', '. . .']:
        return ''
    return token


def remove_noise(tweet_tokens):
    cleaned_tokens = []

    for token in tweet_tokens:

        token = re.sub(
            r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
            " ", token)

        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        cleaned_token = cleaned(token.lower())

        if cleaned_token == "idk":
            cleaned_tokens.append('i')
            cleaned_tokens.append('dont')
            cleaned_tokens.append('know')
            continue
        if cleaned_token == "i'll":
            cleaned_tokens.append('i')
            cleaned_tokens.append('will')
            continue
        if cleaned_token == "you'll":
            cleaned_tokens.append('you')
            cleaned_tokens.append('will')
            continue
        if cleaned_token == "we'll":
            cleaned_tokens.append('we')
            cleaned_tokens.append('will')
            continue
        if cleaned_token == "it'll":
            cleaned_tokens.append('it')
            cleaned_tokens.append('will')
            continue
        if cleaned_token == "it's":
            cleaned_tokens.append('it')
            cleaned_tokens.append('is')
            continue
        if cleaned_token == "i've":
            cleaned_tokens.append('i')
            cleaned_tokens.append('have')
            continue
        if cleaned_token == "you've":
            cleaned_tokens.append('you')
            cleaned_tokens.append('have')
            continue
        if cleaned_token == "we've":
            cleaned_tokens.append('we')
            cleaned_tokens.append('have')
            continue
        if cleaned_token == "they've":
            cleaned_tokens.append('they')
            cleaned_tokens.append('have')
            continue
        if cleaned_token == "you're":
            cleaned_tokens.append('you')
            cleaned_tokens.append('are')
            continue
        if cleaned_token == "we're":
            cleaned_tokens.append('we')
            cleaned_tokens.append('are')
            continue
        if cleaned_token == "they're":
            cleaned_tokens.append('they')
            cleaned_tokens.append('are')
            continue
        if cleaned_token == "let's":
            cleaned_tokens.append('let')
            cleaned_tokens.append('us')
            continue
        if cleaned_token == "she's":
            cleaned_tokens.append('she')
            cleaned_tokens.append('is')
            continue
        if cleaned_token == "he's":
            cleaned_tokens.append('he')
            cleaned_tokens.append('is')
            continue
        if cleaned_token == "that's":
            cleaned_tokens.append('that')
            cleaned_tokens.append('is')
            continue
        if cleaned_token == "i'd":
            cleaned_tokens.append('i')
            cleaned_tokens.append('would')
            continue
        if cleaned_token == "you'd":
            cleaned_tokens.append('you')
            cleaned_tokens.append('would')
            continue
        if cleaned_token == "there's":
            cleaned_tokens.append('there')
            cleaned_tokens.append('is')
            continue
        if cleaned_token == "what's":
            cleaned_tokens.append('what')
            cleaned_tokens.append('is')
            continue
        if cleaned_token == "how's":
            cleaned_tokens.append('how')
            cleaned_tokens.append('is')
            continue
        if cleaned_token == "who's":
            cleaned_tokens.append('who')
            cleaned_tokens.append('is')
            continue
        if cleaned_token == "y'all" or cleaned_token == "ya'll":
            cleaned_tokens.append('you')
            cleaned_tokens.append('all')
            continue

        if cleaned_token.strip() and cleaned_token not in string.punctuation and len(
                cleaned_token) > 2 and cleaned_token not in STOP_WORDS:
            cleaned_tokens.append(cleaned_token)

    return cleaned_tokens


def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding="utf8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('depressionDetector/embeddings/glove.6B.300d.txt')
print("glove file read")
saved_model = tf.keras.models.load_model('depressionDetector/model.h5')
print("model loaded")


def homepage(request):
    return render(request, 'base.html')


def search(request):
    tweet = request.POST.get('tweet_id')
    tweet_no = request.POST.get('tweet_no')
    twitter_id.objects.create(search=tweet, tweets_no=tweet_no)
    
    user = api.get_user(tweet)
    profile_pic = user.profile_image_url_https.replace("normal", "bigger")
    context = {"profile_pic": profile_pic, "name": user.screen_name, "followers": user.followers_count}
    return render(request, 'Twitter/chart.html', context)


class tf_processing(APIView):
    authentication_classes = []
    permission_classes = []

    def get(self, request):
        max_len = 49

        def predict_tweet_sentiment(custom_tweet):
            x_input = sentence_to_indices(remove_noise(tk.tokenize(custom_tweet)), max_len)
 
            return saved_model.predict(np.array([x_input])).item()

        def sentence_to_indices(sentence_words, max_len):
            X = np.zeros((max_len))
            for j, w in enumerate(sentence_words):
                try:
                    index = word_to_index[w]
                except:
                    w = cleaned(w)
                    try:
                        index = word_to_index[w]
                    except:
                        index = word_to_index['unk']
                X[j] = index
            return X
        tweet = twitter_id.objects.all().order_by('-created').first().search
        tweet_no = twitter_id.objects.all().order_by('-created').first().tweets_no
        
        username = tweet
        tweets = api.user_timeline(screen_name=username, count=tweet_no)
        tweet_statement = []
        tweet_rating = []
        tweet_created = []
        for tweet in tweets:
            tweet_statement.append(tweet.text)
            tweet_rating.append(predict_tweet_sentiment(tweet.text))
            tweet_created.append(tweet.created_at.date().strftime("%b %d %Y"))
        data = {
            "tweet": tweet_statement,
            'rating': tweet_rating,
            "created": tweet_created,
            "mean": np.mean(tweet_rating).tolist()
        }
        return Response(data)

















       