import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate
from ast import literal_eval


def getDfMovies(df_mmeta_local):
    """
    Formt das Data Frame entsprechend unserer Bedurfnisse
    :param df_mmeta_local: Eingelesne Datai mit MetaDaten (meist ./data/movies_metadata.csv)
    :return: Pandas DataFrame  mit  vorbereiteteten Daten
    """
    df_movies_local = pd.DataFrame()
    # extract the release year
    df_movies_local['year'] = pd.to_datetime(df_mmeta_local['release_date'], errors='coerce').apply(
        lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

    # extract genres
    df_movies_local['genres'] = df_mmeta_local['genres'].fillna('[]').apply(literal_eval).apply(
        lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    # change the index to movie_id
    df_movies_local['movieId'] = pd.to_numeric(df_mmeta_local['id'])
    df_movies_local = df_movies_local.set_index('movieId')

    # add vote count
    ##FH 2022-11-27 Correction getting Votes from df_mmeta
    df_movies_local['vote_count'] = df_mmeta_local['vote_count']
    df_movies_local['vote_count'] = df_movies_local['vote_count'].astype('int', True, 'ignore')
    ##FH 2022-11-27 Added getting title
    df_movies_local['title'] = df_mmeta_local["title"]
    return df_movies_local


def trainAndValidateCollaboration(df_ratings_local):
    # drop na values
    df_ratings_temp = df_ratings_local.dropna()
    # convert datetime
    df_ratings_temp['timestamp'] = pd.to_datetime(df_ratings_temp['timestamp'], unit='s')

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
    cross_val_results = cross_validate(svd_model_trained, ratings_by_users, measures=['RMSE', 'MAE', 'MSE'], cv=10,
                                       verbose=False)
    test_mae = cross_val_results['test_mae']

    # mean squared errors per fold
    df_test_mae = pd.DataFrame(test_mae, columns=['Mean Absolute Error'])
    df_test_mae.index = np.arange(1, len(df_test_mae) + 1)
    df_test_mae.sort_values(by='Mean Absolute Error', ascending=False).head(15)

    # plot an overview of the performance per fold
    plt.figure(figsize=(6, 4))
    sns.set_theme(style="whitegrid")
    sns.barplot(y='Mean Absolute Error', x=df_test_mae.index, data=df_test_mae, color="b")
    plt.title('Mean Absolute Error')

    return svd_model_trained

def recommend_films_by_collaboration(title, n, user_id, svd_model_trained, df_movies_local):
    """
    Die funktion schlägt dem Nutzer Filme vor, basierend auf dem Film den er eingibt.
    Es wird dabei die Filme aus dem gleichen gewählt die dem Nutzer am besten gefallen werden (laut Prediction)
    :param title: welcher Film ist die Grundlage
    :param n: wie viele Filme sollen vorgeschlagen werden
    :param user_id: für welchen User sollen die Vorschläge sein
    :param svd_model_trained: das Vortrainierte Modell (siehe trainAndValidateCollaboration())
    :param df_movies: die FilmDatenbank
    :return:
    """

    print("Der gewählte Film:")
    chosenMovie = df_movies_local[df_movies_local['title'] == title]
    display(chosenMovie)

    genres = chosenMovie["genres"].iloc[0]
    print(genres)
    print("The variable, genres is of type:", type(genres))

    year = chosenMovie["year"].iloc[0]
    print(year)
    print("The variable, year is of type:", type(year))

    display(df_movies_local[df_movies_local['year'] == year])

    #filter by Year
    df_movies_local = df_movies_local[df_movies_local['year'] == year]


    df_ratings_filtered = df_ratings[df_ratings['userId'] == user_id]

    print(f'number of ratings: {df_ratings_filtered.shape[0]}')

    pred_series = []
    for movie_id, name in zip(df_movies_local.index, df_movies_local['title']):
        # check if the user has already rated a specific movie from the list


        # TODO: HIER ISR DIE VORRAUSSAGE
        rating_pred = svd_model_trained.predict(user_id, movie_id, 0, verbose=False)

        pred_series.append([movie_id, name, rating_pred.est, 0])

    # print the results
    df_recommendations = pd.DataFrame(pred_series, columns=['movieId', 'title', 'predicted_rating', 'actual_rating'])
    display(df_recommendations.sort_values(by='predicted_rating', ascending=False).head(n))


if __name__ == '__main__':
    # in case you have placed the files outside of your working directory, you need to specify a path
    path = 'data/'  # for example: 'data/movie_recommendations/'

    # load the movie metadata
    df_moviesmetadata = pd.read_csv(path + 'movies_metadata.csv', low_memory=False)
    # remove invalid records with invalid ids
    df_mmeta = df_moviesmetadata.drop([19730, 29503, 35587])
    # load the movie ratings
    df_ratings = pd.read_csv(path + 'ratings_small.csv', low_memory=False)

    df_movies = getDfMovies(df_mmeta)
    svd_model_trained = trainAndValidateCollaboration(df_ratings)

    title = 'The Dark Knight Rises'
    user_id = 400
    n = 10

    recommend_films_by_collaboration(title, n, user_id, svd_model_trained, df_movies)


