# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:24:30 2020

@author: nkraj
"""
import pandas as pd

####### preprocessing
movies = pd.read_csv('movies_new.csv')
ratings = pd.read_csv('ratings_new.csv')

# turn Film-Noir and Sci-Fi into lowercase one-word
movies['genres'] = [x.replace('Film-Noir', 'filmnoir') for x in movies['genres']]
movies['genres'] = [x.replace('Sci-Fi', 'scifi') for x in movies['genres']]

# drop time stamp from ratings
ratings = ratings.drop('timestamp', axis=1)

# create a combimned dataframe of ratings and movies
data = pd.merge(movies, ratings).drop(['genres'], axis=1)

# save only ratings on movies with more than 15 ratings
data = data.groupby('movieId').filter(lambda x: x['rating'].count() >= 15)

# create data frame for model based approaches
model_data = data.drop('title', axis=1)[['userId','movieId','rating']] 

# keep only movies that are in model_data
movies = movies[movies['movieId'].isin(list(model_data.movieId.unique()))]

# export processed data
movies.to_csv('movies_cleaned.csv', index=False)
model_data.to_csv('model_data.csv', index=False)
data.to_csv('combined_data.csv', index=False)