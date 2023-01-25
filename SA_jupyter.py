#!/usr/bin/env python
# coding: utf-8

# # 1. Collecting Reviews
# 

# In[2]:


import requests
from bs4 import BeautifulSoup
import re

r = requests.get('https://www.yelp.com/biz/wingz-it-iz-alsip-10')  #<-- base link

# https://www.yelp.com/biz/wingz-it-iz-alsip-10?start=30 is page 4


soup = BeautifulSoup(r.text, 'html.parser')

#grab all reviews where the tag is span
#  where lang = en and class inculdes raw__
#


regex = re.compile('raw__')
results = soup.find_all('span', {'lang':'en'}, class_= regex)

reviews = [review.text for review in results]
#identify the pattern in the links
['https://www.yelp.com/biz/wingz-it-iz-alsip-10' # page 1,
'https://www.yelp.com/biz/wingz-it-iz-alsip-10?start=30' # page 4
]



# ## Scraping Multiple Pages

# In[4]:


#dynamic link generation
links = [f'https://www.yelp.com/biz/wingz-it-iz-alsip-10?start={10+x*10}' for x in range(3)]
links.insert(0,'https://www.yelp.com/biz/wingz-it-iz-alsip-10')


# In[5]:


reviews = []
for link in links:
    r = requests.get(link)
    soup = BeautifulSoup(r.text, 'html.parser')
    regex = re.compile('raw__')
    results = soup.find_all('span', {'lang':'en'}, class_= regex)
    reviews = [*reviews, *[result.text for result in results]]
reviews


# # 2. Analysing the Reviews

# In[6]:


import pandas as pd
import numpy as np


# In[10]:


df = pd.DataFrame(np.array(reviews), columns = ['review'])


# In[11]:


df.head()


# ## Calculating Text Metrics

# In[22]:


df['word_count'] = df['review'].apply(lambda x: len(str(x).split(' ')))
df['char_count'] = df['review'].str.len()


# In[23]:


df.head()


# In[21]:


print(df[df['word_count']==438]['review'].values[0])


# ## Counting Stopwords

# In[27]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# In[31]:


stop_words = stopwords.words('english')


# In[32]:


df['stopword_count'] = df['review'].apply(lambda x: len([x for x in x.split() if x in stop_words]))


# In[34]:


df.describe()


# # 3. Cleaning the Dataset

# ## Lowercasing all words

# In[35]:


df['review_lower'] = df['review'].apply(lambda x: ' '.join(x.lower() for x in x.split()))


# In[36]:


df.head()


# ## Stripping Punctuation

# In[37]:


df['review_nopunc'] = df['review_lower'].str.replace('[^\w\s]','')


# In[38]:


df.head()


# ## Removing Stopwords

# In[41]:


df['review_nostop'] = df['review_nopunc'].apply(lambda x: ' '.join(x for x in x.split() if x not in stop_words))


# In[42]:


df.head()


# ## Visualizing Common Words

# In[43]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[71]:


freq = pd.Series(" ".join(df['review_nostop']).split()).value_counts()[:80].reset_index()


# In[72]:


freq.columns = ['words','frequencies']


# In[ ]:





# In[75]:


plt.figure(figsize = (15,4))
plt.title('Common Words')

sns.barplot(x = 'words', y = 'frequencies', data = freq)
plt.xticks(rotation =90)
plt.show()


# ## Stripping out Common words

# In[78]:


freq.tail(50)


# In[94]:


other_stopwords = ['5', 'know', 'explained', 'two', 'could', 'go', 'thing', 'im', 'got', 'the', 'u', 'say', 'literally', 'say'
                  'where', 'my', 'they', 'a', 'i', 'me', 'and']


# In[95]:


df['no_other'] = df['review_nopunc'].apply(lambda x: ' '.join(x for x in x.split() if x not in other_stopwords))


# df.head()

# ## 4. LEMMATIZE REVIEWS

# In[99]:


from textblob import Word
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[100]:


df['cleaned_review'] = df['no_other'].apply(lambda x: ' '.join(Word(word).lemmatize() for word in x.split()))


# In[101]:


print(df['review'].iloc[0])
print('~~~~~~~~~~~~~~~~~~~~~~~~~')
print(df['cleaned_review'].iloc[0])


# ## 5. SENTIMENT ANALYSIS

# In[103]:


from textblob import TextBlob


# In[104]:


df['polarity'] = df['cleaned_review'].apply(lambda x: TextBlob(x).sentiment[0])
df['subjectivity'] = df['cleaned_review'].apply(lambda x: TextBlob(x).sentiment[1])


# In[109]:


print(df.iloc[7]['review'])


# In[ ]:




