
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.DataFrame({
    'title': ['The Matrix','Inception','Interstellar','The Dark Knight','Avengers: Endgame'],
    'genres': ['Action|Sci-Fi','Action|Sci-Fi|Thriller','Adventure|Sci-Fi|Drama','Action|Crime|Drama','Action|Adventure|Sci-Fi']
})

cv = CountVectorizer(stop_words='english')
matrix = cv.fit_transform(movies['genres'])
similarity = cosine_similarity(matrix)

st.title("Movie Recommendation System")
selected_movie = st.selectbox("Select a movie:", movies['title'].tolist())

def recommend(movie):
    idx = movies[movies['title']==movie].index[0]
    distances = similarity[idx]
    top5 = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    return [movies.iloc[i[0]].title for i in top5]

if st.button("Recommend"):
    recs = recommend(selected_movie)
    st.subheader("Top 5 Recommendations:")
    for r in recs:
        st.write(r)
