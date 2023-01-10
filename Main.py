# Data processing
import pandas as pd
import numpy as np
import scipy.stats

# # Visualization
import seaborn as sns

# # Similarity
from sklearn.metrics.pairwise import cosine_similarity

ratings=pd.read_csv('ml-latest-small/rating-menu.csv')


# Read in data
movies = pd.read_csv('ml-latest-small/menus.csv')# Take a look at the data

# Merge ratings and movies datasets
df = pd.merge(ratings, movies, on='idMenu', how='inner')# Take a look at the data


# Aggregate by movie
agg_ratings = df.groupby('idPelanggan').agg(mean_rating = ('rating', 'mean'),
                                                number_of_ratings = ('rating', 'count')).reset_index()# Keep the movies with over 100 ratings

agg_ratings_GT100 = agg_ratings[agg_ratings['number_of_ratings']>0]

# Check the information of the dataframe

# # Check popular movies
agg_ratings_GT100.sort_values(by='number_of_ratings', ascending=False).head()

# # # # Visulization
# # sns.jointplot(x='mean_rating', y='number_of_ratings', data=agg_ratings_GT100)

# # Merge data
df_GT100 = pd.merge(df, agg_ratings_GT100[['idPelanggan']], on='idPelanggan', how='inner')



# # Create user-item matrix
matrix = df_GT100.pivot_table(index='idMenu', columns='idPelanggan', values='rating')


# # # Normalize user-item matrix
matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 0)


# # # Item similarity matrix using Pearson correlation
# item_similarity = matrix.T.corr()



# # # # # Item similarity matrix using cosine similarity
item_similarity_cosine = cosine_similarity(matrix_norm.fillna(0))
print(item_similarity_cosine)

# # # # Pick a user ID
# picked_idPelanggan = 1

# # # Pick a movie
# picked_movie = 'Ayam Goreng Kampung'

# # # Movies that the target user has watched
# picked_idPelanggan_watched = pd.DataFrame(matrix_norm[picked_idPelanggan].dropna(axis=0, how='all')\
#                           .sort_values(ascending=False))\
#                           .reset_index()\
#                           .rename(columns={1:'rating'})
# picked_idPelanggan_watched.head()

# # Similarity score of the movie American Pie with all the other movies
# picked_movie_similarity_score = item_similarity[[picked_movie]].reset_index().rename(columns={'Ayam Goreng Kampung':'similarity_score'})# Rank the similarities between the movies user 1 rated and American Pie.
# n = 5

# print(picked_movie_similarity_score)

# picked_idPelanggan_watched_similarity = pd.merge(left=picked_idPelanggan_watched, 
#                                             right=picked_movie_similarity_score, 
#                                             on='nameMenu', 
#                                             how='inner')\
#                                      .sort_values('similarity_score', ascending=False)[:5]

# # # Calculate the predicted rating using weighted average of similarity scores and the ratings from user 1
# predicted_rating = round(np.average(picked_idPelanggan_watched_similarity['rating'], 
#                                     weights=picked_idPelanggan_watched_similarity['similarity_score']), 6)


# # Item-based recommendation function
# def item_based_rec(picked_idPelanggan=1, number_of_similar_items=5, number_of_recommendations =3):
#   import operator
#   # Movies that the target user has not watched

#   picked_idPelanggan_unwatched = pd.DataFrame(matrix_norm[picked_idPelanggan].isna()).reset_index()
#   picked_idPelanggan_unwatched = picked_idPelanggan_unwatched[picked_idPelanggan_unwatched[1]==True]['idMenu'].values.tolist()  # Movies that the target user has watched
#   picked_idPelanggan_watched = pd.DataFrame(matrix_norm[picked_idPelanggan].dropna(axis=0, how='all')\
#                             .sort_values(ascending=False))\
#                             .reset_index()\
#                             .rename(columns={1:'rating'})

  
#   # Dictionary to save the unwatched movie and predicted rating pair
#   rating_prediction ={}    # Loop through unwatched movies          
#   for picked_movie in picked_idPelanggan_unwatched: 
#     # Calculate the similarity score of the picked movie iwth other movies
#     picked_movie_similarity_score = item_similarity[[picked_movie]].reset_index().rename(columns={picked_movie:'similarity_score'})
#     # Rank the similarities between the picked user watched movie and the picked unwatched movie.
#     picked_idPelanggan_watched_similarity = pd.merge(left=picked_idPelanggan_watched, 
#                                                 right=picked_movie_similarity_score, 
#                                                 on='idMenu', 
#                                                 how='inner')\
#                                         .sort_values('similarity_score', ascending=False)[:number_of_similar_items]
#     # Calculate the predicted rating using weighted average of similarity scores and the ratings from user 1
#     predicted_rating = round(np.average(picked_idPelanggan_watched_similarity['rating'], 
#                                         weights=picked_idPelanggan_watched_similarity['similarity_score']), 6)

#     # Save the predicted rating in the dictionary
#     rating_prediction[picked_movie] = predicted_rating
#     # Return the top recommended movies
#   return sorted(rating_prediction.items(), key=operator.itemgetter(1), reverse=True)[:number_of_recommendations]

# # Get recommendations
# recommended_movie = item_based_rec(picked_idPelanggan=1, number_of_similar_items=3, number_of_recommendations =1)
# print(recommended_movie)
