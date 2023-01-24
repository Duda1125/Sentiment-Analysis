
#IMPORT DEPENDENCIES
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import Word
from textblob import TextBlob


def get_reviews():

#dynamic link generation
    links = [f'https://www.yelp.com/biz/wingz-it-iz-alsip-10?start={10+x*10}' for x in range(3)]
    links.insert(0,'https://www.yelp.com/biz/wingz-it-iz-alsip-10')




    reviews = []
    for link in links:
        r = requests.get(link)
        soup = BeautifulSoup(r.text, 'html.parser')
        regex = re.compile('raw__')
        results = soup.find_all('span', {'lang':'en'}, class_= regex)
        reviews = [*reviews, *[result.text for result in results]]
    return reviews


def preprocess(reviews):


    df = pd.DataFrame(np.array(reviews), columns = ['review'])

    stop_words = stopwords.words('english')


    df['review_lower'] = df['review'].apply(lambda x: ' '.join(x.lower() for x in x.split()))


    df['review_nopunc'] = df['review_lower'].str.replace('[^\w\s]','') #stripped puncuation


    df['review_nostop'] = df['review_nopunc'].apply(lambda x: ' '.join(x for x in x.split() if x not in stop_words)) #removes stop words

    other_stopwords = ['5', 'know', 'explained', 'two', 'could', 'go', 'thing', 'im', 'got', 'the', 'u', 'say', 'literally', 'say'
                  'where', 'my', 'they', 'a', 'i', 'me', 'and']



    df['no_other'] = df['review_nopunc'].apply(lambda x: ' '.join(x for x in x.split() if x not in other_stopwords))









    df['cleaned_review'] = df['no_other'].apply(lambda x: ' '.join(Word(word).lemmatize() for word in x.split()))

    return df


def calculate_sentiment(df):

    df['polarity'] = df['cleaned_review'].apply(lambda x: TextBlob(x).sentiment[0])
    df['subjectivity'] = df['cleaned_review'].apply(lambda x: TextBlob(x).sentiment[1])
    #return final data frame

    return df




if __name__ == "__main__":
    reviews = get_reviews()
    df = preprocess(reviews)
    sentiment_df = calculate_sentiment(df)
    sentiment_df.to_csv('results.csv')