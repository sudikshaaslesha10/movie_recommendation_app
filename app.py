import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")

st.title("Movie Recommendation System")

cv = CountVectorizer(stop_words="english")
matrix = cv.fit_transform(movies["genres"])
similarity = cosine_similarity(matrix)

selected_movie = st.selectbox("Select a movie", movies["title"].values)

def recommend(movie):
    index = movies[movies["title"] == movie].index[0]
    distances = similarity[index]

    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

    recommended_movies = []
    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    for movie in recommendations:
        st.write(movie)
