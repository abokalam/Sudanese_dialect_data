import GetOldTweets3 as got
import pandas as pd

import os.path
from os import path

def hashtag_tweets(query, count, begin_date, end_date):
    # Creation of query object
    tweetCriteria = got.manager.TweetCriteria().setUsername(query)\
                                           .setSince(begin_date)\
                                           .setUntil(end_date)\
                                            .setMaxTweets(count)
    # Creation of list that contains all tweets
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)

    # Creating list of chosen tweet data
    user_tweets = [[tweet.date, tweet.text, tweet.username] for tweet in tweets]
    
    print("tweets list created")

    # Creation of dataframe from tweets list
    tweets_df = pd.DataFrame(user_tweets, columns = ['Datetime', 'Text', 'username'])

    print("dataframe created")
    
    to_drop = []

    for index, row in tweets_df.iterrows():
        if len(row['Text']) < 20:
          to_drop.append(index)

          
    tweets_df.drop(to_drop,  inplace = True)

    print("dataframe cleaned")
    print("from: " + begin_date + "   until: " + end_date)
    print("tweets saved with length: " + str(len(tweets_df)))
    tweets_df.to_csv('{}-tweets.csv'.format(query), mode='a', sep=',')
    

    return tweets_df