
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Sample Movie Dataset (~30 movies)
# -----------------------------
movies = pd.DataFrame({
    'title': [
        'The Matrix', 'Inception', 'Interstellar', 'The Dark Knight', 'Avengers: Endgame',
        'Titanic', 'The Notebook', 'Jurassic Park', 'The Lion King', 'Toy Story',
        'Avatar', 'The Avengers', 'Gladiator', 'Pirates of the Caribbean', 'Finding Nemo',
        'Shrek', 'Frozen', 'Mad Max: Fury Road', 'Black Panther', 'Guardians of the Galaxy',
        'Harry Potter', 'The Hobbit', 'The Lord of the Rings', 'Star Wars', 'The Godfather',
        'Pulp Fiction', 'Fight Club', 'The Shawshank Redemption', 'Forrest Gump', 'Jumanji'
    ],
    'genres': [
        'Action|Sci-Fi','Action|Sci-Fi|Thriller','Adventure|Sci-Fi|Drama','Action|Crime|Drama',
        'Action|Adventure|Sci-Fi','Romance|Drama','Romance|Drama','Adventure|Sci-Fi',
        'Animation|Adventure|Family','Animation|Adventure|Comedy',
        'Action|Adventure|Fantasy','Action|Adventure|Sci-Fi','Action|Drama|Historical',
        'Adventure|Action|Fantasy','Animation|Adventure|Family','Animation|Comedy|Family',
        'Animation|Adventure|Family','Action|Adventure|Sci-Fi','Action|Adventure|Superhero',
        'Action|Adventure|Comedy','Adventure|Fantasy|Family','Adventure|Fantasy','Adventure|Fantasy|Action',
        'Action|Adventure|Sci-Fi','Crime|Drama','Crime|Thriller','Drama|Crime','Drama|Romance','Adventure|Family|Fantasy'
    ]
})

# -----------------------------
# Vectorize genres
# -----------------------------
cv = CountVectorizer(stop_words='english')
matrix = cv.fit_transform(movies['genres'])
similarity = cosine_similarity(matrix)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Movie Recommendation System")
selected_movie = st.selectbox("Select a movie", movies['title'].tolist())

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    top_movies = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in top_movies]

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    st.subheader("Top 5 Recommendations:")
    for m in recommendations:
        st.write(m)
