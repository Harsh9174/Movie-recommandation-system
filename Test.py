import streamlit as st
import pandas as pd
import requests
import clickhouse_connect
import re
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# ClickHouse connection details
def get_clickhouse_client():
    client = clickhouse_connect.get_client(
        host='nxefycxt62.eastus2.azure.clickhouse.cloud',
        port=8443,
        username='default',
        password='ZoNOGOZPSv61~'
    )
    return client

client = get_clickhouse_client()

# Setup the database and table
try:
    client.command("CREATE DATABASE IF NOT EXISTS Movies")
    client.command("USE Movies")
except clickhouse_connect.driver.exceptions.DatabaseError as e:
    st.error(f"Database setup error: {e}")

query = "SELECT * FROM Movies.Final;"
try:
    Movies_Combined_final = pd.DataFrame(client.query(query).result_rows, columns=["Movie_ID", "Title", "Tags"])
except clickhouse_connect.driver.exceptions.DatabaseError as e:
    st.error(f"Query execution error: {e}")
    st.stop()
    



# Preprocess tags
CV = CountVectorizer(max_features=5000, stop_words='english')
Vector = CV.fit_transform(Movies_Combined_final['Tags']).toarray()
PS = PorterStemmer()

def stem(text):
    return " ".join(PS.stem(word) for word in text.split())

Movies_Combined_final['Tags'] = Movies_Combined_final['Tags'].apply(stem)
Similarity = cosine_similarity(Vector)

# Function to fetch movie poster
def fetch_poster(movie_id):
    res = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US')
    data = res.json()
    return f"https://image.tmdb.org/t/p/original/{data['poster_path']}"

# Recommendation function
def recommend(movie):
    movie_index = Movies_Combined_final[Movies_Combined_final['Title'] == movie].index[0]
    distances = Similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]
    
    recommended_movies = []
    recommended_posters = []
    for i in movie_list:
        movie_id = Movies_Combined_final.iloc[i[0]].Movie_ID
        recommended_movies.append(Movies_Combined_final.iloc[i[0]].Title)
        recommended_posters.append(fetch_poster(movie_id))
    return recommended_movies, recommended_posters

# Streamlit UI
def model_page():
    st.title("Movie Recommendation System")

    options = st.selectbox("Choose Your Favourite Movie", Movies_Combined_final['Title'])

    if st.button('Recommend'):
        recommendations, posters = recommend(options)
        cols = st.columns(5)
        for col, rec, poster in zip(cols, recommendations, posters):
            col.text(rec)
            col.image(poster)

def movie_entry():
    st.title('Movie Data Entry')

    client = get_clickhouse_client()

    def insert_data(movie_id, title, tags):
        query = "INSERT INTO Movies.Final (Movie_ID, Title, Tags) VALUES"
        data = f"({movie_id}, '{title}', '{tags}')"
        client.command(query + data)

    movie_id = st.number_input('Movie ID', min_value=1, step=1)
    title = st.text_input('Movie Title')
    overview = st.text_input('Description')
    keywords = st.text_input('Keywords')
    genres = st.text_input('Genres')
    cast = st.text_input('Actor/Actress')
    director = st.text_input('Director')

    tags = " ".join(
        [tag.replace(' ', '') for tag in re.split(r'[,\s]+', overview + " " + keywords + " " + genres + " " + cast + " " + director) if tag.strip()]
    ).lower()

    if st.button('Submit'):
        insert_data(movie_id, title, tags)
        st.success('Data inserted successfully!')

# Streamlit multi-page setup
PAGES = {
    "Data Entry": movie_entry,
    "Movie Recommendation System": model_page
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page()

