# Movie Recommender: Project Overview
- Created an SVD recommendation system to recommend movies to a user with an MSE of 0.78.
- Used MovieLens ratings data (link below).
- Preprocessed the data by mergin datasets and filtering for movies with a minimum of 15 ratings.
- Created functions for user-based, item-based, KNN, and SVD recommender systems.
- Tested all systems to find the best performing based on MSE.
- Created a bonus content-based recommender system based on genre.

## Code and Resources Used
**Python Version:** 3.7\
**Packages:** pandas, numpy, sklearn, surprise, collections, \
**Recommender Systems Articles (in no particular order):** 
- https://towardsdatascience.com/intro-to-recommender-system-collaborative-filtering-64a238194a26
- https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0
- https://towardsdatascience.com/the-4-recommendation-engines-that-can-predict-your-movie-tastes-109dc4e10c52
- https://www.ethanrosenthal.com/2015/11/02/intro-to-collaborative-filtering/

## Data Sources
MovieLens dataset - https://grouplens.org/datasets/movielens/

## Recommendation Systems
For this project I built 4 different recommendation systems to suggest movies to the user. I built a "bonus" content-based recommender system as well. I call it a bonus system because it can not be evalutated on its own. It simply recommends movies of the same or similar genre.\
**Collaborative Filtering**
1. User-based: compared users to other users based on their historical ratings
2. Item-based:compared movies to other movies based on their historical ratings
3. KNN with Means: can be user or item based but takes into account the mean ratings; used the surprise python package
**Matrix Factorization**
4. Singular Value Decompostion(SVD): leveraged the SVD model from the surprise python package; found optimal hyperparameters via GridSearchCV
**Content-Based**
5. TF-IDF: compared titles using their genre in a TF-IDF algorithm; leveraged sklearn's TfdifVectorizer

## Recommenddation Performance
I compared the models performance using mean squared error (MSE). I felt as though this was the correct metric since the rating scale is relatively small (1 to 5). Therefore, I needed to exacerbate the weight of incorrect guesses so a poor prediction algorithm would be very apparent.

                    user       item       knn       svd
            MSE  13.453443  13.011294  0.888318  0.784149
