import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate
from ast import literal_eval

if __name__ == '__main__':
    # in case you have placed the files outside of your working directory, you need to specify a path
    path = 'data/'  # for example: 'data/movie_recommendations/'

    # load the movie metadata
    df_moviesmetadata = pd.read_csv(path + 'movies_metadata.csv', low_memory=False)
    print(df_moviesmetadata.shape)
    print(df_moviesmetadata.columns)
    df_moviesmetadata.head(1)


    # load the movie ratings
    df_ratings=pd.read_csv(path + 'ratings.csv', low_memory=False)

    print(df_ratings.shape)
    print(df_ratings.columns)
    df_ratings.head(3)

    rankings_count = df_ratings.rating.value_counts().sort_values()
    sns.barplot(x=rankings_count.index.sort_values(), y=rankings_count, color="b")
    sns.set_theme(style="whitegrid")


    # remove invalid records with invalid ids
    df_mmeta = df_moviesmetadata.drop([19730, 29503, 35587])

    df_movies = pd.DataFrame()

    # extract the release year
    df_movies['year'] = pd.to_datetime(df_mmeta['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

    # extract genres
    df_movies['genres'] = df_mmeta['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    # change the index to movie_id
    df_movies['movieId'] = pd.to_numeric(df_mmeta['id'])
    df_movies = df_movies.set_index('movieId')

    # add vote count
    ##FH 2022-11-27 Correction getting Votes from df_mmeta
    df_movies['vote_count'] = df_mmeta['vote_count']
    df_movies['vote_count'] =df_movies['vote_count'].astype('int',True,'ignore')
    ##FH 2022-11-27 Added getting title
    df_movies['title'] = df_mmeta["title"]
    # drop na values
    df_ratings_temp = df_ratings.dropna()

    # convert datetime
    df_ratings_temp['timestamp'] = pd. to_datetime(df_ratings_temp['timestamp'], unit='s')

    print(f'unique users: {len(df_ratings_temp.userId.unique())}, ratings: {len(df_ratings_temp)}')
    df_ratings_temp.head()


    # The Reader class is used to parse a file containing ratings.
    # The file is assumed to specify only one rating per line, such as in the df_ratings_temp file above.
    reader = Reader()
    ratings_by_users = Dataset.load_from_df(df_ratings_temp[['userId', 'movieId', 'rating']], reader)

    # Split the Data into train and test
    train_df, test_df = train_test_split(ratings_by_users, test_size=.2)

    # train an SVD model
    svd_model = SVD()
    svd_model_trained = svd_model.fit(train_df)

    # 10-fold cross validation
    cross_val_results = cross_validate(svd_model_trained, ratings_by_users, measures=['RMSE', 'MAE', 'MSE'], cv=10, verbose=False)
    test_mae = cross_val_results['test_mae']

    # mean squared errors per fold
    df_test_mae = pd.DataFrame(test_mae, columns=['Mean Absolute Error'])
    df_test_mae.index = np.arange(1, len(df_test_mae) + 1)
    df_test_mae.sort_values(by='Mean Absolute Error', ascending=False).head(15)

    # plot an overview of the performance per fold
    plt.figure(figsize=(6,4))
    sns.set_theme(style="whitegrid")
    sns.barplot(y='Mean Absolute Error', x=df_test_mae.index, data=df_test_mae, color="b")
    # plt.title('Mean Absolute Error')


    # predict ratings for a single user_id and for all movies
    user_id = 400 # some test user from the ratings file

    # create the predictions
    pred_series= []
    df_ratings_filtered = df_ratings[df_ratings['userId'] == user_id]

    print(f'number of ratings: {df_ratings_filtered.shape[0]}')
    #TODO: DAS IST DIE LOOP
    for movie_id, name in zip(df_movies.index, df_movies['title']):
        # check if the user has already rated a specific movie from the list
        rating_real = df_ratings.query(f'movieId == {movie_id}')['rating'].values[0] if movie_id in df_ratings_filtered['movieId'].values else 0
        # generate the prediction


        #TODO: HIER ISR DIE VORRAUSSAGE
        rating_pred = svd_model_trained.predict(user_id, movie_id, rating_real, verbose=False)
        # add the prediction to the list of predictions
        pred_series.append([movie_id, name, rating_pred.est, rating_real])

    # print the results
    df_recommendations = pd.DataFrame(pred_series, columns=['movieId', 'title', 'predicted_rating', 'actual_rating'])
    df_recommendations.sort_values(by='predicted_rating', ascending=False).head(15)


    # predict ratings for the combination of user_id and movie_id
    user_id = 217 # some test user from the ratings file
    movie_id = 4002
    rating_real = df_ratings.query(f'movieId == {movie_id} & userId == {user_id}')['rating'].values[0]
    movie_title = df_movies[df_movies.index == 862]['title'].values[0]

    print(f'Movie title: {movie_title}')
    print(f'Actual rating: {rating_real}')

    # predict and show the result
    rating_pred = svd_model_trained.predict(user_id, movie_id, rating_real, verbose=True)

    tfidf = TfidfVectorizer(stop_words='english')
    df_moviesmetadata['overview'] = df_moviesmetadata['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(df_moviesmetadata['overview'])
    print(tfidf_matrix.shape)

    cosine_matrix_description = linear_kernel(tfidf_matrix, tfidf_matrix)

    index_mapping = pd.Series(df_moviesmetadata.index, index=df_moviesmetadata['title']).drop_duplicates()


def getRatingsByUsers(path):
    df_ratings = pd.read_csv(path)
    df_ratings_temp = df_ratings.dropna()

    # convert datetime
    df_ratings_temp['timestamp'] = pd.to_datetime(df_ratings_temp['timestamp'], unit='s')

    print(f'unique users: {len(df_ratings_temp.userId.unique())}, ratings: {len(df_ratings_temp)}')
    df_ratings_temp.head()

    # The Reader class is used to parse a file containing ratings.
    # The file is assumed to specify only one rating per line, such as in the df_ratings_temp file above.
    reader = Reader()
    return Dataset.load_from_df(df_ratings_temp[['userId', 'movieId', 'rating']], reader)


def BuildKiModel(ratings_by_users):
    # Split the Data into train and test
    train_df, test_df = train_test_split(ratings_by_users, test_size=.2)

    # train an SVD model
    svd_model = SVD()
    return svd_model.fit(train_df)


def calculateMSNE(path):
    ratings_by_users = getRatingsByUsers(path)
    svd_model_trained = BuildKiModel(ratings_by_users)

    cross_val_results = cross_validate(svd_model_trained, ratings_by_users, measures=['RMSE', 'MAE', 'MSE'], cv=10,
                                       verbose=False)

#%%
tfidf = TfidfVectorizer(stop_words='english')
df_moviesmetadata['overview'] = df_moviesmetadata['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df_moviesmetadata['overview'])
cosine_matrix_description = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, cosine_matrix_description=cosine_matrix_description):
    # Get the index of the movie that matches the title
    idx = index_mapping[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_matrix_description[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df_moviesmetadata['title'].iloc[movie_indices]


def getMovieSet(df_mmeta):
    df_movies = pd.DataFrame()

    # extract the release year
    df_movies['year'] = pd.to_datetime(df_mmeta['release_date'], errors='coerce').apply(
        lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

    # extract genres
    df_movies['genres'] = df_mmeta['genres'].fillna('[]').apply(literal_eval).apply(
        lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    # change the index to movie_id
    df_movies['movieId'] = pd.to_numeric(df_mmeta['id'])
    df_movies = df_movies.set_index('movieId')

    # add vote count
    ##FH 2022-11-27 Correction getting Votes from df_mmeta
    df_movies['vote_count'] = df_mmeta['vote_count']
    df_movies['vote_count'] = df_movies['vote_count'].astype('int', True, 'ignore')
    ##FH 2022-11-27 Added getting title
    df_movies['title'] = df_mmeta["title"]
    return df_movies


def recommend_films_by_collaboration(title, n, user_id, method):
    # finde ähnliche Filme anhand von
    # a --> jahr (+/- 5) und Genre
    # b --> Methode aus Teil 1
    movieSet = None
    if method == 'a':
        tfidf = TfidfVectorizer(stop_words='english')
        df_moviesmetadata['overview'] = df_moviesmetadata['overview'].fillna('')
        tfidf_matrix = tfidf.fit_transform(df_moviesmetadata['overview'])
        print(tfidf_matrix.shape)

        cosine_matrix_description = linear_kernel(tfidf_matrix, tfidf_matrix)

        index_mapping = pd.Series(df_moviesmetadata.index, index=df_moviesmetadata['title']).drop_duplicates()
        movieSet = get_recommendations(title, cosine_matrix_description)

        print(movieSet)

    # bereche alle Scores für diese Filme

    # Sortire nach vorraussichtlicher score

    # BRECHNE RSME


if __name__ == '__main__':

    recommend_films_by_collaboration("Goldfinger", 15, 400, 'a')
