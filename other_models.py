# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 12:22:38 2020

@author: nkraj
"""
import numpy as np
import pandas as pd

# for genre recs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# for user-based algorithm
from sklearn.metrics.pairwise import cosine_similarity

# libraries for model based approaches
from surprise import KNNWithMeans, SVD, Dataset, Reader
from collections import defaultdict

# import data
movies = pd.read_csv('movies_cleaned.csv')
data = pd.read_csv('combined_data.csv')
model_data = pd.read_csv('model_data.csv')

# create a movie title dictionary
movie_dict = movies.drop('genres',axis=1).set_index('movieId').to_dict()

# take a smaller subset of the model_data
model_data_og = model_data.copy()
model_data = model_data[model_data['userId'] < 500]

# take a smaller subset of data dataframe
data_small = data[data['userId'] < 500]

###### build model-based algorithms
reader = Reader(rating_scale=(1,5))
model_df = Dataset.load_from_df(model_data, reader)
trainset = model_df.build_full_trainset()
testset = trainset.build_anti_testset()

### KNN-based
# Use user_based true/false to switch between user-based or item-based collaborative filtering
knn_algo = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': True})
knn_algo.fit(trainset)
# run the trained model on testset
knn_pred = knn_algo.test(testset)

# now map the predictions to each user
knn_preds_list = defaultdict(list)
for uid, iid, true_r, est, _ in knn_pred:
    knn_preds_list[uid].append((iid,est))

# next sort the predictions by user
for uid, user_ratings in knn_preds_list.items():
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    knn_preds_list[uid] = user_ratings

###### variables needed for recommendation functions
# needed for genre recs
tfv = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=1)
x = tfv.fit_transform(movies['genres'])
cosine_sim = linear_kernel(x, x)
    
def get_genre_recs(title, n_recs=5):
    """
    Parameters
    ----------
    title : string
        A string matching the title of a movie in the 'title' column of the 
        movies dataframe. This will be the movie to base recommendations off
        of.

    Returns
    -------
    rec_movies: list
        A list of the recommended movies similar to the entered title
        based on genre

    """
    idx = list(movies[movies['title'] == title].index)[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores]
    rec_movies = movies['title'].iloc[movie_indices].drop(idx).head(n_recs).to_list()
    return rec_movies

###### build recommender functions
def get_title_recs(title, n_recs=5):
    """
    Parameters
    ----------
    title : string
        title : string
        A string matching the title of a movie in the 'title' column of the 
        movies dataframe. This will be the movie to base recommendations off
        of using an item-based approach.
    n_recs : int, optional
        Number of recommendations to return. The default is 5.

    Returns
    -------
    rec_movies : pandas series
        A list of the recommended movies similar to the entered title
        based on an item-based approach.

    """
    item_pivot_table = data_small.pivot_table(index = ["userId"],
                                        columns = ["title"],
                                        values = "rating",
                                        fill_value=0)
    movie_watched = item_pivot_table[title]
    sim_scores = item_pivot_table.corrwith(movie_watched)
    sim_scores = sim_scores.sort_values(ascending=False)
    rec_movies = sim_scores.drop(title).index[:n_recs].to_list()
    return rec_movies

def user_based_recs(user, n_recs=5, n_users=5):
    """
    Parameters
    ----------
    user : int
        UserId number that corresponds to a user in the data (or ratings) table.
    n_recs : int, optional
        Number of recommendations to return. The default is 5.
    n_users : int, optional
        Number of similar users to find and use in weighted rankings. The 
        default is 5.

    Returns
    -------
    rec_movies : pandas series
        A list of the recommended movies similar to the entered title
        based on an item-based approach.

    """
    # create pivot
    user_pivot_table = data.pivot_table(index = ["userId"],
                                        columns = ["title"],
                                        values = "rating",
                                        fill_value=0)
    # calculate the cosine similarity scores for the given user
    # subtract one from user due to indexing mismatch
    user_scores = pd.Series(cosine_similarity(user_pivot_table)[user-1]) 
    # match indexing (increase by one to match pivot table)
    user_scores.index = range(1,len(user_scores)+1)
    # take top n similar users
    sim_users_scores = user_scores.sort_values(ascending=False)[1:n_users+1]
    # save top n userId's as list
    sim_users = list(sim_users_scores.index) 
    # create user-item matrix based on top n users
    sim_users_df = user_pivot_table.T[sim_users].T 
    # take the weighted sum of similar user's rankings
    user_recs =  sim_users_scores.dot(sim_users_df) / np.array([np.abs(sim_users_scores).sum(axis=0)]) 
    # remove the titles the user has already seen
    liked_items = list(data[data['userId'] == user].sort_values('rating', ascending=False)['title'])
    liked_removed = [x for x in list(user_recs.index) if x not in liked_items]
    rec_movies = user_recs.loc[liked_removed].sort_values(ascending=False).index[:n_recs].to_list()
    return rec_movies

def knn_based_recs(user, n_recs=3):
    # retrive the top n recommendations    
    temp = knn_preds_list.get(user)[:n_recs]
    rec_movies = pd.Series([x[0] for x in temp]).map(movie_dict['title']).to_list()    
    return rec_movies

def run_svd_model(trainset, testset):
    svd_model = SVD(n_factors=150, n_epochs=20, lr_all=0.008, 
                reg_all=0.1, random_state=24)
    svd_model.fit(trainset)
    # predict ratings for all user-item pairs that are not in the trainset
    svd_preds = svd_model.test(testset)
    
    svd_preds_list = defaultdict(list)
    for uid, iid, true_r, est, _ in svd_preds:
        svd_preds_list[uid].append((iid,est))
    
    # next sort the predictions by user
    for uid, user_ratings in svd_preds_list.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        svd_preds_list[uid] = user_ratings

    return svd_preds_list

svd_preds_list = run_svd_model(trainset, testset)

def svd_based_recs(user, n_recs):
    temp = svd_preds_list.get(user)[:n_recs]
    rec_movies = pd.Series([x[0] for x in temp]).map(movie_dict['title']).to_list()
    
    return rec_movies

def get_recs(user=None, title=None, n_recs=5, kind='genre'):
    if kind == 'genre':
        if title:
            # get genre recs
            genre_recs = get_genre_recs(title, n_recs)
            print('='*50)
            print('Genre Based Recommendations for {}'.format(title))
            print('='*50)
            for i in range(len(genre_recs)):
                print('{}. {}'.format(i+1, genre_recs[i]))
        else: print('Please provide a genre.')
    elif kind == 'model':
        if user:
            svd_recs = svd_based_recs(user, n_recs)
            print('='*50)
            print('Model Based Recommendations for userId {}'.format(user))
            print('='*50)
            for i in range(len(svd_recs)):
                print('{}. {}'.format(i+1, svd_recs[i]))
        else: print('Please provide a userId.')
    else: print('Please provide an appropriate input for kind.')