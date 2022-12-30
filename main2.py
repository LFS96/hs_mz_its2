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
    Diese Funktion bereitet einen DataFrame von Filmmetadata auf. Sie nimmt einen DataFrame mit allen Filmmetadata (df_mmeta_local) als Eingabe und gibt einen neuen, formatierten DataFrame (df_movies_local) zurück.
    Zunächst wird aus dem Release-Datum des Films das Erscheinungsjahr extrahiert und in einer neuen Spalte year im DataFrame gespeichert. Wenn das Release-Datum fehlt oder nicht konvertiert werden kann, wird der Wert NaN verwendet.
    Anschließend wird die Spalte genres des Eingabedatensatzes extrahiert und auf ihren Inhalt (Liste von Genres) transformiert. Die Spalte genres wird dann in eine Liste von Genrenamen konvertiert.
    Der Index des DataFrames wird dann auf die Spalte movieId geändert und auf numerische Werte konvertiert.
    Danach wird die Spalte vote_count aus dem Eingabedatensatz extrahiert und in den neuen DataFrame eingefügt. Der Datentyp von vote_count wird auf Integer konvertiert.
    Schließlich wird auch der Titel des Films in den neuen DataFrame eingefügt.
    Der neu formatierte DataFrame wird schließlich zurückgegeben.

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
    """
    Diese Funktion trainiert und validiert ein Collaborative Filtering-Modell für Vorhersage von Bewertungen von Filmen durch Benutzer. Der Code nutzt das Python-Modul Surprise für Collaborative Filtering.
    Zunächst werden fehlende Werte (NaN) aus dem Eingabedatensatz entfernt. Danach wird der Zeitstempel in der Spalte timestamp in ein datetime-Format konvertiert.
    Anschließend wird das Reader-Modul von Surprise verwendet, um den Datensatz zu analysieren und in eine Dataset-Instanz zu laden. Der Datensatz enthält Benutzer-ID, Film-ID und Bewertung in Spalten userId, movieId und rating.
    Der Datensatz wird dann in einen Trainings- und einen Testsatz aufgeteilt. Das Verhältnis wird durch den Wert von test_size festgelegt, in diesem Fall 20%.
    Danach wird ein SVD-Modell (Singular Value Decomposition) von Surprise instanziiert und mit dem Trainingsdatensatz trainiert.
    Schließlich wird das trainierte Modell mithilfe der Funktion cross_validate von Surprise validiert. Dabei wird der komplette Datensatz in 10 Fälle aufgeteilt und das Modell 10-fach über alle Fälle validiert. Als Messgrößen werden der Root Mean Squared Error (RMSE), der Mean Absolute Error (MAE) und der Mean Squared Error (MSE) berechnet.
    Die Funktion gibt schließlich das trainierte und validierte SVD-Modell zurück. Zusätzlich wird ein Balkendiagramm erstellt, das den MAE für jeden Durchlauf der Cross-Validation anzeigt.
    :param df_ratings_local:
    :return:
    """
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
    Diese Funktion empfiehlt Filme für einen gegebenen Benutzer basierend auf Vorhersagen des Bewertungen mithilfe eines Collaborative Filtering-Modells und den von diesem Benutzer bereits bewerteten Filmen.
    Zunächst wird der gewählte Film angezeigt, indem der DataFrame df_movies_local nach dem angegebenen Titel durchsucht wird. Der Film wird dann anhand seines Erscheinungsjahrs gefiltert und der DataFrame df_movies_local wird auf Filme aus dem gleichen Jahr beschränkt.
    Danach wird der DataFrame df_ratings nach Bewertungen des angegebenen Benutzers gefiltert und die Anzahl der Bewertungen wird ausgegeben.
    Schließlich wird für jeden Film in df_movies_local eine Vorhersage der Bewertung durch den Benutzer mithilfe des übergebenen, trainierten Collaborative Filtering-Modells berechnet. Die Vorhersagen werden in einem DataFrame df_recommendations zusammengefasst, der die Filme mit ihren vorhergesagten und tatsächlichen Bewertungen enthält. Schließlich wird der DataFrame sortiert und die empfohlenen Filme werden anhand der vorhergesagten Bewertungen absteigend angezeigt. Die Anzahl der empfohlenen Filme wird durch den Wert von n festgelegt.


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
    year = chosenMovie["year"].iloc[0]
    display(df_movies_local[df_movies_local['year'] == year])

    #filter by Year
    df_movies_local = df_movies_local[df_movies_local['year'] == year]

    df_ratings_filtered = df_ratings[df_ratings['userId'] == user_id]

    print(f'number of ratings: {df_ratings_filtered.shape[0]}')

    pred_series = []
    for movie_id, name in zip(df_movies_local.index, df_movies_local['title']):
        rating_pred = svd_model_trained.predict(user_id, movie_id, 0, verbose=False)
        pred_series.append([movie_id, name, rating_pred.est, 0])
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


