import numpy as np
import pandas as pd

# libraries for model based approaches
from surprise import SVD, Dataset, Reader
from collections import defaultdict

# import data
movies = pd.read_csv('movies_cleaned.csv')
data = pd.read_csv('combined_data.csv')
model_data = pd.read_csv('model_data.csv')

# create a movie title dictionary
movie_dict = movies.drop('genres',axis=1).set_index('movieId').to_dict()

# take a smaller subset of the model_data
model_data_og = model_data.copy()
model_data = model_data[model_data['userId'] < 1000]
    

###### build model-based algorithms
def generate_train_test_data(model_data):
    reader = Reader(rating_scale=(1,5))
    model_df = Dataset.load_from_df(model_data, reader)
    trainset = model_df.build_full_trainset()
    testset = trainset.build_anti_testset()
    return trainset, testset

trainset, testset = generate_train_test_data(model_data)
### SVD model
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

###### user input
# generate a sample of n movies to be ranked by the user
def  list_to_rank(movies, n_titles=150, use_all=True):
    if use_all:
        movie_list = movies.drop('genres', axis=1)
    else:
        movie_list = data.groupby(['movieId', 'title']).agg({'rating': np.mean})
        movie_list = movie_list[movie_list['rating'] >= 3]
    if n_titles > len(movies):
        print('Choose fewer samples than total movies')
    movie_list = movie_list.sample(n_titles)
    movie_list['rating'] = ''
    movie_list.to_csv('rate_these.csv', index=False)
    return print('Movie list created. Ready for ratings.')

def integrate_new_ratings(model_data):
    # read in user inputs and drop nulls
    user_ratings = pd.read_csv('rate_these.csv').dropna(axis=0)
    userId_idx = model_data['userId'].max()+1
    user_ratings['userId'] = userId_idx
    user_ratings = user_ratings.drop('title', axis=1)
    user_ratings = user_ratings[model_data.columns.to_list()]
    model_data = model_data.append(user_ratings)
    # update the train and test sets
    trainset, testset = generate_train_test_data(model_data)
    # update svd model and predictions
    svd_preds_list = run_svd_model(trainset, testset)
    return model_data, trainset, testset, svd_preds_list

### run below if adding a new user
# model_data, trainset, testset, svd_preds_list = integrate_new_ratings(model_data)
