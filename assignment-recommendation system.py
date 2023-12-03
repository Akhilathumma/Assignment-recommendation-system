# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 09:58:48 2023

@author: user
"""

import pandas as pd
books=pd.read_csv('D:\\Data science\\book.csv',encoding='latin1')
books
list(books)
books.rename({'Unnamed: 0':'index','User.ID':'user_id','Book.Title':'book_title','Book.Rating':'book_rating'},axis=1,inplace=True)
books.set_index('index', inplace=True)
books

books['user_id'].unique()
print('number of unique users are:',format(len(books['user_id'].unique())))
print('number of unique books:',format(len(books['book_title'].unique())))
print('number of unique ratings:',format(len(books['book_rating'].unique())))
books['book_rating'].value_counts()
books.info()
books.describe()
books.book_rating.describe()

books.isnull().any()
books.isnull().sum()
books.duplicated()
books[books.duplicated()].shape
books[books.duplicated()]

books.groupby('book_title')['book_rating'].mean().sort_values(ascending=False).head()
books.groupby('book_title')['book_rating'].count().sort_values(ascending=False).head()

ratings=pd.DataFrame(books.groupby('book_title')['book_rating'].mean())
ratings

ratings['num of ratings']=pd.DataFrame(books.groupby('book_title')['book_rating'].count())
ratings.head()

import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)

plt.figure(figsize=(10,4))
ratings['book_rating'].hist(bins=70)

import seaborn as sns
sns.jointplot(x='book_rating',y='num of ratings',data=ratings,alpha=0.5)

top_books=books['book_title'].value_counts().head()
top_books

user_books_df = books.pivot_table(index='user_id',columns = 'book_title', values = 'book_rating').fillna(0)
user_books_df

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation, jaccard
ratings.sort_values('num of ratings',ascending=False).head(10)

stardust_user_ratings = user_books_df['Stardust']
fahrenheit_user_rating = user_books_df['Fahrenheit 451']
fahrenheit_user_rating.head()

similar_to_fahrenheit = user_books_df.corrwith(fahrenheit_user_rating)
similar_to_stardust = user_books_df.corrwith(stardust_user_ratings)
corr_fahrenheit = pd.DataFrame(similar_to_fahrenheit,columns=['Correlation'])
corr_fahrenheit.dropna(inplace=True)
corr_fahrenheit.head()

corr_fahrenheit.sort_values('Correlation',ascending=False).head(10)

corr_fahrenheit = corr_fahrenheit.join(ratings['book_rating'])
corr_fahrenheit.head()

corr_fahrenheit[corr_fahrenheit['book_rating']>5].sort_values('Correlation',ascending=False).head()

corr_stardust = pd.DataFrame(similar_to_stardust,columns=['Correlation'])
corr_stardust.dropna(inplace=True)
corr_stardust = corr_stardust.join(ratings['num of ratings'])
corr_stardust[corr_stardust['num of ratings']>4].sort_values('Correlation',ascending=False).head()

user_books_df.head()

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation, jaccard
user_sim = 1 - pairwise_distances(user_books_df.values, metric = 'cosine')
user_sim

user_sim_df = pd.DataFrame(user_sim)
user_sim_df
user_sim_df.iloc[:5,:5]
import numpy as np
np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:5,0:5]

user_sim_df.index = list(user_books_df.index)
user_sim_df.columns = list(user_books_df.index)
user_sim_df

user_id_eight = user_sim_df.sort_values([9], ascending=False).head(100)
user_id_eight[9]

books[(books['user_id']==8) | (books['user_id']==14)]

user_sim_df.idxmax(axis=1)

books[(books['user_id']==8) | (books['user_id']==14)]

def give_reco(customer_id):
    tem = list(user_sim_df.sort_values([customer_id],ascending=False).head(100).index)
    #print('similar customer ids:',tem)
    movie_list=[]
    for i in tem:
        movie_list=movie_list+list(books[books['user_id']==i]['book_title'])
    #print('Common movies within customer',movie_list)
    return set(movie_list)-set(books[books['user_id']==customer_id]['book_title'])
give_reco(14)
