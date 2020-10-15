import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

# for genre recs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# libraries for model based approaches
from surprise import KNNWithMeans, SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, GridSearchCV

movies = pd.read_csv('movies_cleaned.csv')
data = pd.read_csv('combined_data.csv')
model_data = pd.read_csv('model_data.csv')

# take a smaller sample of data
data_og = data.copy()
data = data[data['userId'] < 500]
model_data_og = model_data.copy()
model_data = model_data[model_data['userId'] < 500]


####### Optimize Parameters

# create a param grid for optimizing the SVD model
reader = Reader(rating_scale=(1, 5))
gs_data = Dataset.load_from_df(model_data,reader)
param_grid = {'n_factors': [50, 100, 150],
              'n_epochs': [5, 10, 20], 
              'lr_all': [0.002, 0.005, 0.008],
              'reg_all': [0.02, 0.1, 0.4],
              'random_state': [24]}
gs = GridSearchCV(SVD, param_grid, measures=['mse'], cv=3)
gs.fit(gs_data)
# best RMSE score
print(gs.best_score['mse'])
# combination of parameters that gave the best RMSE score
print(gs.best_params['mse'])

# best params yields:
# {'n_factors': 150, 'n_epochs': 20, 'lr_all': 0.008, 
# 'reg_all': 0.1, 'random_state': 24}

####### define functions
# define a train_test_split function
def my_train_test_split(data):
    """
    Parameters
    ----------
    data : dataframe
        Should be a pandas dataframe with columns userId, movieId, and
        rating (in that order).

    Returns
    -------
    train : dataframe
        A training set that has removed 10 ratings per user.
    test : dataframe
        A test set consisting of 10 ratings per user.

    """
    n_users = data.userId.nunique()
    test = pd.DataFrame(columns=['userId','movieId','rating'])
    train = data.copy()

    for user in range(1,n_users+1):
        temp = data[(data['userId'] == user) & (data['rating'] > 0)].sample(10, random_state=24)
        test = pd.concat([test, temp])
    train = pd.merge(train, test, left_index=True, right_index=True, how="outer", indicator=True).query('_merge=="left_only"')
    train = train[['userId_x','movieId_x','rating_x']]
    train = train.rename(columns={
        'userId_x': 'userId', 
        'movieId_x':'movieId', 
        'rating_x':'rating'
        })
    return train, test

# define a function for mse
def get_mse(pred, actual):
    pred = pred[actual.nonzero()] #ignore zeros
    actual = actual[actual.nonzero()] #ignore zeros
    return mean_squared_error(pred, actual)


# create function for testing the performance of memory based recommenders
def test_memory_based(data, kind='item'):
    train, test = my_train_test_split(data)
    temp = train.pivot_table(index = ["userId"],columns = ["movieId"],
                             values = "rating",fill_value=0).to_numpy()
    actual = test.pivot_table(index = ["userId"],columns = ["movieId"],
                              values = "rating",fill_value=0).to_numpy()
    if kind == 'item':
        sim = temp.T.dot(temp) + 1e-9 # add small value to fix divide by zero
        pred = temp.dot(sim) / np.array([np.abs(sim).sum(axis=1)])
    elif kind == 'user':
        sim = temp.dot(temp.T) + 1e-9 # add small value to fix divide by zero
        pred = sim.dot(temp) / np.array([np.abs(sim).sum(axis=1)]).T
    else:
        print('Please choose an approriate input for kind.')
        return
    test_mse = get_mse(pred, actual)
    return test_mse

def test_knn_based(data):
    """
    Parameters
    ----------
    data : dataframe
        Dataframe with columns userId, movieId, and rating in that order.

    Returns
    -------
    test_mse : float
        The mean squared error for the knn based algorithm.

    """
    reader = Reader(rating_scale=(1, 5))
    knn_data = Dataset.load_from_df(data,reader)
    trainset, testset = train_test_split(knn_data, test_size=.10, random_state=24)
    algo = KNNWithMeans(k=5, sim_options={
        'name': 'pearson_baseline', 
        'user_based': True
        })
    algo.fit(trainset)
    predictions = algo.test(testset)
    test_mse = accuracy.mse(predictions, verbose=False)
    return test_mse

def test_svd(data):
    reader = Reader(rating_scale=(1, 5))
    svd_data = Dataset.load_from_df(data,reader)
    trainset, testset = train_test_split(svd_data, test_size=.10, random_state=24)
    svd_model = SVD(n_factors=150, n_epochs=20, lr_all=0.008, 
                    reg_all=0.1, random_state=24)
    svd_model.fit(trainset)
    predictions = svd_model.test(testset)
    test_mse = accuracy.mse(predictions, verbose=False)
    return test_mse

# create df to track performance
performance_table = pd.DataFrame(index=['MSE'], columns=['user', 'item'])


####### testing

# test memory based algorithms
for kind in ['user', 'item']:
    performance_table[kind] = test_memory_based(model_data, kind=kind)
    print('{}-based MSE: {:0.4f}'.format(kind, performance_table[kind][0]))
    
# test knn based algorithm
knn_mse = test_knn_based(model_data)
performance_table['knn'] = knn_mse
print('KNN Model MSE: {}'.format(knn_mse))

# test SVD algorithm
svd_mse = test_svd(model_data)
performance_table['svd'] = svd_mse
print('SVD Model MSE: {}'.format(svd_mse))

performance_table
