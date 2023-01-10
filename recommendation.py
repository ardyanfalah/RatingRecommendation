# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the ratings data
ratings=pd.read_csv('ml-latest-small/ratings.csv')

# Create a pivot table of the ratings data
rating_pivot = ratings.pivot_table(index='userId', columns='movieId', values='rating')

# Replace missing values with 0
rating_pivot = rating_pivot.fillna(0)
# Create a function to calculate the similarity between two items using cosine similarity
def calculate_similarity(item1, item2):
  # Create a matrix of the ratings for the two items
  item_ratings = pd.concat([item1, item2], axis=1)
  
  # Replace missing values with the mean rating for each item
  item_ratings = item_ratings.fillna(item_ratings.mean())
  
  # Calculate the similarity using cosine similarity
  item_similarity = cosine_similarity(item_ratings.T)[0,1]
  return item_similarity

# Create a function to generate recommendations for a user
def recommend(user_id, num_recommendations):
  # Get the ratings for the user
  user_ratings = rating_pivot[rating_pivot.index == user_id]
  
  # Create an empty list to store the recommended items
  recommendations = []
  
  # Loop through each item the user has not rated
  for item_id in rating_pivot.columns:
    if item_id not in user_ratings.columns:
      # Calculate the similarity between the current item and all other items
      item_similarities = rating_pivot.apply(lambda x: calculate_similarity(x, user_ratings[item_id]), axis=0)
      
      # Sort the items by similarity
      item_similarities = item_similarities.sort_values(ascending=False)
      
      # Add the item with the highest similarity to the recommendations list
      recommendations.append((item_id, item_similarities[0]))
      
  # Sort the recommendations by similarity
  recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
  
  # Return the top N recommendations
  return recommendations[:num_recommendations]

# Test the recommendation function
print(calculate_similarity(3,2))
