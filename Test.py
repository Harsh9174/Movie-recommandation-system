import streamlit as st
import pandas as pd
import pickle
import requests
import clickhouse_connect
import re
import ast
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import sys
import json
import os

CLICKHOUSE_CLOUD_HOSTNAME = os.getenv('CLICKHOUSE_CLOUD_HOSTNAME', 'nxefycxt62.eastus2.azure.clickhouse.cloud')
CLICKHOUSE_CLOUD_USER = os.getenv('CLICKHOUSE_CLOUD_USER', 'default')
CLICKHOUSE_CLOUD_PASSWORD = os.getenv('CLICKHOUSE_CLOUD_PASSWORD', 'ZoNOGOZPSv61~')

def get_clickhouse_client():
    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_CLOUD_HOSTNAME,
        port=8443,
        username=CLICKHOUSE_CLOUD_USER,
        password=CLICKHOUSE_CLOUD_PASSWORD
    )
    return client

client = get_clickhouse_client()

# Setup the database and table
try:
    client.command("CREATE DATABASE IF NOT EXISTS Movies")
    client.command("USE Movies")
except clickhouse_connect.driver.exceptions.DatabaseError as e:
    st.error(f"Database setup error: {e}")

# Fetch the data
query = "SELECT * FROM Movies.Final"
try:
    result = client.query(query)
    Movies_Combined_final = pd.DataFrame(result.result_rows, columns=["Movie_ID", "Title", "Tags"])
except clickhouse_connect.driver.exceptions.DatabaseError as e:
    st.error(f"Query execution error: {e}")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error: {e}")
    st.stop()

CV = CountVectorizer(max_features=5000,stop_words='english')

Vector = CV.fit_transform(Movies_Combined_final['Tags']).toarray()

PS = PorterStemmer()
def stem(text):
    L = []
    for i in text.split():
        L.append(PS.stem(i))
    return " ".join(L) 

Movies_Combined_final['Tags'] = Movies_Combined_final['Tags'].apply(stem)

Similarity = cosine_similarity(Vector)




# Function for the model page
def model_page():
# Load your DataFrame
    def fetch_poster(movie_id):
        res = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&lanfuage=en-US'.format(movie_id))
        data = res.json()
        return "https://image.tmdb.org/t/p/original/" + data['poster_path']

    def recommend(Movie):
        Movie_index = Movies_Combined_final[Movies_Combined_final['Title'] == Movie].index[0]
        distance = Similarity[Movie_index]
        Movie_List = sorted(list(enumerate(distance)),reverse=True,key=lambda x : x[1])[1:6]

        recommended_Movies = []
        recommended_Movies_poster = []
        for i in Movie_List:
            Movie_ID = Movies_Combined_final.iloc[i[0]].Movie_ID
            recommended_Movies.append(Movies_Combined_final.iloc[i[0]].Title)
            recommended_Movies_poster.append(fetch_poster(Movie_ID))
        return recommended_Movies,recommended_Movies_poster



# Title at the top
    st.title("Movie Recommendation System")

# Sidebar with movie selection
    options = st.selectbox(
        "Choose Your Favourite Movie",
        Movies_Combined_final['Title']
    )

    if st.button('Recommend'):
        recommandations,poster = recommend(options)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(recommandations[0])
            st.image(poster[0])

        with col2:
            st.text(recommandations[1])
            st.image(poster[1])

        with col3:
            st.text(recommandations[2])
            st.image(poster[2])

        with col4:
            st.text(recommandations[3])
            st.image(poster[3])

        with col5:
            st.text(recommandations[4])
            st.image(poster[4])


# Function for the data insertion page
def Movie_Entry():
    

# ClickHouse connection detail

    def insert_data(Movie_ID,Title,Tags):
        query = """
            INSERT INTO Movies.Final (Movie_ID,Title, Tags) VALUES
        """
        data = f"({Movie_ID},'{Title}', '{Tags}')"
        client.command(query + data)

# Streamlit UI
    st.title('Movie Data Entry')
    M_id = st.number_input('Movie ID',min_value=1,step=1)
    Title = st.text_input('Movie Title')
    Overview = st.text_input('Discription')
    Keywords = st.text_input('Keywords')
    Genres = st.text_input('Genres')
    Cast = st.text_input('Actor/Actoress')
    Director = st.text_input('Director')

    overview = [tag.strip() for tag in re.split(r'[,\s]+', Overview) if tag.strip()]
    keywords = [tag.strip() for tag in re.split(r'[,\s]+', Keywords) if tag.strip()]
    genres = [tag.replace(' ', '') for tag in re.split(r'[,\s]+', Genres) if tag.strip()]
    cast = [tag.strip() for tag in re.split(r'[,\s]+', Cast) if tag.strip()]
    director = [tag.strip() for tag in re.split(r'[,\s]+', Director) if tag.strip()]

# Combine all tags into one list
    tags = overview + genres + keywords + cast + director

    tags = " ".join(tags).lower()



    if st.button('Submit'):
        insert_data(M_id,Title,tags)
        st.success('Data inserted successfully!')

# Streamlit multi-page setup
PAGES = {
    "Data Entry": Movie_Entry,
    "Movie Recommandation System": model_page
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]
page()


