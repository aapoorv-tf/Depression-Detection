
import tweepy as tw
import twitter_credentials
import csv
import pandas as pd



consumer_key = twitter_credentials.API_KEY
consumer_secret = twitter_credentials.API_KEY_SECRET
access_key = twitter_credentials.ACCESS_TOKEN
access_secret = twitter_credentials.ACCESS_TOKEN_SECRET
  
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tw.API(auth, wait_on_rate_limit=True)
tweets_text = []
for tweet in tw.Cursor(api.search,
                           q = "depression",
#                            since = "2021-01-14",
#                             until = "2021-01-15",
                           lang = "en").items(10000):
    tweets_text.append([tweet.user.screen_name,tweet.user.location,tweet.text])
    
df_Tweet = pd.DataFrame(tweets_text)
df_Tweet.to_csv('depression.csv', header=["User_Name","Location","Tweets"], index=False,)
