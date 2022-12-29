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

def getMovieSet(df_mmeta):
    df_mmeta = df_mmeta.drop([19730, 29503, 35587])
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

def getRatingsByUsers(path):
    df_ratings = pd.read_csv('data/ratings.csv', low_memory=False)
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


def recommend_films_by_collaboration(title, n, user_id, method):
    # finde ähnliche Filme anhand von
    # a --> jahr (+/- 5) und Genre
    # b --> Methode aus Teil 1
    movieSet = None
    if method == 'a':
        tfidf = TfidfVectorizer(stop_words='english')
        metadata['overview'] = metadata['overview'].fillna('')
        tfidf_matrix = tfidf.fit_transform(metadata['overview'])
        print(tfidf_matrix.shape)

        cosine_matrix_description = linear_kernel(tfidf_matrix, tfidf_matrix)

        index_mapping = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()
        movieSet = get_recommendations(title, cosine_matrix_description)

        print(movieSet)

    # df_movies = getMovieSet()match movieSet(Z60)

    movieSet = getMovieSet(metadata)

    svd_model_trained = BuildKiModel(getRatingsByUsers('data/movies_metadata.csv'))
    df_ratings = pd.read_csv('data/ratings.csv', low_memory=False)
    df_ratings_filtered = df_ratings[df_ratings['userId'] == user_id]
    for movie_id, name in zip(movieSet.index, movieSet['title']):
        rating_real = df_ratings.query(f'movieId == {movie_id}')['rating'].values[0] if movie_id in df_ratings_filtered[
            'movieId'].values else 0
        rating_pred = svd_model_trained.predict(user_id, indices, rating_real, verbose=False)
        print('rating ' + rating_pred + " film:" + movie_id)

    # Sortire nach vorraussichtlicher score

    # BRECHNE RSME

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v / (v + m) * R) + (m / (m + v) * C)

    # Function that takes in movie title as input and outputs most similar movies

def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    # Return empty list in case of missing/malformed data
    return []

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

if __name__ == '__main__':

    # Load Movies Metadata
    metadata = pd.read_csv('data/movies_metadata.csv', low_memory=False)

    # Print the first three rows
    print(metadata.head(3))
    C = metadata['vote_average'].mean()
    print(C)

    # Calculate the minimum number of votes required to be in the chart, m
    m = metadata['vote_count'].quantile(0.90)
    print(m)

    # Filter out all qualified movies into a new DataFrame
    q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
    print(q_movies.shape)

    # Define a new feature 'score' and calculate its value with `weighted_rating()
    q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
    print(metadata.shape)

    # Sort movies based on score calculated above
    q_movies = q_movies.sort_values('score', ascending=False)

    # Print the top 15 movies
    print( q_movies[['title', 'vote_count', 'vote_average', 'score']].head(20))

    #Print plot overviews of the first 5 movies.
    print(metadata['overview'].head())

    # Import TfIdfVectorizer from scikit-learn
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')

    # Replace NaN with an empty string
    metadata['overview'] = metadata['overview'].fillna('')

    # Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(metadata['overview'])

    # Output the shape of tfidf_matrix
    print(tfidf_matrix.shape)

    # Array mapping from feature integer indices to feature name.
    print(tfidf.get_feature_names()[5000:5010])

    # Import linear_kernel
    from sklearn.metrics.pairwise import linear_kernel

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    print(cosine_sim.shape)
    print(cosine_sim[1])

    # Construct a reverse map of indices and movie titles
    indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()
    print(indices[:10])

    print(get_recommendations('The Dark Knight Rises'))
    print(get_recommendations('The Godfather'))

    # Load keywords and credits
    credits = pd.read_csv('data/credits.csv')
    keywords = pd.read_csv('data/keywords.csv')

    # Remove rows with bad IDs.
    metadata = metadata.drop([19730, 29503, 35587])

    # Convert IDs to int. Required for merging
    keywords['id'] = keywords['id'].astype('int')
    credits['id'] = credits['id'].astype('int')
    metadata['id'] = metadata['id'].astype('int')

    # Merge keywords and credits into your main metadata dataframe
    metadata = metadata.merge(credits, on='id')
    metadata = metadata.merge(keywords, on='id')

    # Print the first two movies of your newly merged metadata
    print(metadata.head(2))

    # Parse the stringified features into their corresponding python objects
    from ast import literal_eval

    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        metadata[feature] = metadata[feature].apply(literal_eval)

    # Import Numpy
    import numpy as np

    def get_director(x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

    def get_list(x):
        if isinstance(x, list):
            names = [i['name'] for i in x]
            # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
            if len(names) > 3:
                names = names[:3]
            return names

        # Return empty list in case of missing/malformed data
        return []

    # Define new director, cast, genres and keywords features that are in a suitable form.
    metadata['director'] = metadata['crew'].apply(get_director)

    features = ['cast', 'keywords', 'genres']
    for feature in features:
        metadata[feature] = metadata[feature].apply(get_list)

    # Print the new features of the first 3 films
    print(metadata[['title', 'cast', 'director', 'keywords', 'genres']].head(3))

    # Function to convert all strings to lower case and strip names of spaces
    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            # Check if director exists. If not, return empty string
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''

    # Apply clean_data function to your features.
    features = ['cast', 'keywords', 'director', 'genres']

    for feature in features:
        metadata[feature] = metadata[feature].apply(clean_data)

    def create_soup(x):
        return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

    # Create a new soup feature
    metadata['soup'] = metadata.apply(create_soup, axis=1)

    print(metadata[['soup']].head(2))

    # Import CountVectorizer and create the count matrix
    from sklearn.feature_extraction.text import CountVectorizer

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(metadata['soup'])

    print(count_matrix.shape)

    # Compute the Cosine Similarity matrix based on the count_matrix
    from sklearn.metrics.pairwise import cosine_similarity

    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

    # Reset index of your main DataFrame and construct reverse mapping as before
    metadata = metadata.reset_index()
    indices = pd.Series(metadata.index, index=metadata['title'])

    print(get_recommendations('The Dark Knight Rises', cosine_sim2))
    print(get_recommendations('The Godfather', cosine_sim2))

    print(recommend_films_by_collaboration('The Dark Knight Rises', 15, 400, 'a'))
