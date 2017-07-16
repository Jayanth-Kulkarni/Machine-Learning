import tweepy
from textblob import TextBlob



consumer_key = "pETUHqDPZhbbcLCVeZRU0FIXW"
consumer_secret = "5cJPF8XVhTfPSVJBbs56CZe4bWgrxzsQeV8ievxFswGOSSfP1n"

access_token = "1027116877-TCWU7XZPYR0ZavOIu3GEcC2pwH2jCrzOExQj3KR"
access_token_secret = "56aqGYclGBWVpu8GJgIsfzOZpQ71J97ga0aTbxnBaGrEm"

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Trump')

for tweet in public_tweets:
    print tweet.text
    analysis = TextBlob(tweet.text)
    print analysis.sentiment , "\n\n"

# Polarity measures how positive or negative the tweet is
# subjectivity is a measure of how much of an opinion it is to how factual it is
