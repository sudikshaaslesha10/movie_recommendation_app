
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset
movies = pd.DataFrame({
    'title': ['The Matrix','Inception','Interstellar','The Dark Knight','Avengers: Endgame',
              'Titanic','The Notebook','Jurassic Park','The Lion King','Toy Story'],
    'genres': ['Action|Sci-Fi','Action|Sci-Fi|Thriller','Adventure|Sci-Fi|Drama',
               'Action|Crime|Drama','Action|Adventure|Sci-Fi','Romance|Drama','Romance|Drama',
               'Adventure|Sci-Fi','Animation|Adventure|Family','Animation|Adventure|Comedy']
})

# Vectorize genres
cv = CountVectorizer(stop_words='english')
matrix = cv.fit_transform(movies['genres'])
similarity = cosine_similarity(matrix)

# Streamlit UI
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
