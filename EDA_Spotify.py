#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


sns.set_style("darkgrid")

#loading dataset
df = pd.read_csv("data_spotify.csv")
df.head()


# In[8]:


df.info()


# In[12]:


#statistical summary
df.describe()


# In[14]:


#Data Analysis
five_most_popular = df.groupby("artist").count().sort_values(by = 'song_title', ascending = False)["song_title"][:5]
five_most_popular


# In[15]:


five_most_popular.plot.barh(color = "green")
plt.show


# In[22]:


# Most Loudest Track

most_loudest_tracks = df[["loudness", "song_title"]].sort_values(by = "loudness", ascending = True)[:5]
most_loudest_tracks


# In[23]:


plt.figure(figsize = (12, 7))
sns.barplot(x = "loudness", y = "song_title", data = most_loudest_tracks)
plt.title("Top 5 loudest Tracks")
plt.show()


# In[26]:


#Artist having most dancing songs

dance = df[['danceability', "song_title", "artist"]].sort_values(by = "danceability", ascending = False)[: 5]
dance


# In[28]:


plt.figure(figsize = (12, 7))
sns.barplot(x = 'danceability', y = 'artist', data = dance)
plt.title("Artist having most dancing songs")
plt.show()


# In[29]:


top_ten_inst = df[["instrumentalness", 'song_title', "artist" ]].sort_values(by = 'instrumentalness', ascending = False)[:5]
top_ten_inst


# In[36]:


plt.figure(figsize = (12,7))
sns.barplot(x = "instrumentalness", y = "artist", data = top_ten_inst)
plt.title("Top Ten Instrumental Songs")
plt.show()


# In[38]:


plt.figure(figsize =(12,7))
plt.pie(x ="instrumentalness", data = top_ten_inst, autopct = '%1.2f%%',labels = top_ten_inst.song_title)


# In[43]:


int_feature_col = ["tempo", "loudness", "acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "liveness", "speechiness", "valence"]


# In[48]:


for col in int_feature_col:
    pos = df[df['target'] == 1][col]
    neg = df[df['target'] == 0][col]
    
    plt.figure(figsize=(12,7))
    sns.distplot(pos, bins=20, label="Positive", color = "green")
    sns.distplot(neg, bins = 20, label = "Negative", color = "red")
    plt.title(f"Histogram plot for {col}")
    plt.legend(loc = "upper right")
    plt.show()

