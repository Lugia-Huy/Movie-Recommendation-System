import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

###Import data
films = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
#print(films.head())
#print(ratings.head())

###Make Utility Matrix
final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')
final_dataset.fillna(0,inplace=True)
#print(final_dataset.head())

###Making the necessary modifications as per the threshold set. film has over 10 user voted, user voted at least 50 films
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_films_voted = ratings.groupby('userId')['rating'].agg('count')

final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index,:]
final_dataset = final_dataset.loc[:,no_films_voted[no_films_voted > 50].index]
#print(final_dataset)

###Removing sparsity
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)
#print(final_dataset)
final_dataset.to_csv('basedata.csv', index=False)

###Recommendation function
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)
#print(csr_data)

def get_movie_recommendation(movie_name):
    total_films_recommend = 10
    movie_list = films[films['title'].str.contains(movie_name)]  
    if len(movie_list):        
        film_index= movie_list.iloc[0]['movieId']
        film_index = final_dataset[final_dataset['movieId'] == film_index].index[0]
        distances , indices = knn.kneighbors(csr_data[film_index],n_neighbors=total_films_recommend+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            film_index = final_dataset.iloc[val[0]]['movieId']
            idx = films[films['movieId'] == film_index].index
            recommend_frame.append({'Title':films.iloc[idx]['title'].values[0]})
        df = pd.DataFrame(recommend_frame,index=range(1,total_films_recommend+1))
        return df
    else:
        return "Your film's name is not exist in data, so please check your input again!"

'''
###evaluate

'''
###------------------###

name_movie = input("Enter name's movie you like: ")
films = get_movie_recommendation(name_movie)
print("Maybe you will like:")
print(films)

