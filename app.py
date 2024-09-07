import streamlit as st
import pickle
import pandas as pd
import requests


# Function to fetch movie poster using TMDB API
def fetch_poster(movie_id):
    # API call to get movie details by movie ID
    response = requests.get(
        'https://api.themoviedb.org/3/movie/{}?api_key=ce44761597b02356081101edd08a8c6b&language=en-US'.format(movie_id)
    )
    data = response.json()  # Parse the JSON response
    # Return the full URL to the poster image
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']


# Function to recommend movies based on similarity
def recommend(movie):
    # Get the index of the selected movie in the dataset
    movie_index = movies[movies['title'] == movie].index[0]
    # Get the similarity scores for the selected movie
    distances = similarity[movie_index]
    # Sort movies based on similarity scores, ignoring the first one (the selected movie itself)
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    # Lists to store recommended movie titles and their posters
    recommended_movies = []
    recommended_movies_posters = []

    # Loop through the top 5 most similar movies
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id  # Get the movie ID
        recommended_movies.append(movies.iloc[i[0]].title)  # Get the movie title
        recommended_movies_posters.append(fetch_poster(movie_id))  # Get the poster using the API

    return recommended_movies, recommended_movies_posters  # Return both titles and posters


# Load the movie dictionary (metadata) and similarity matrix from pickle files
movie_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movie_dict)  # Convert the movie dictionary to a DataFrame

similarity = pickle.load(open('similarity.pkl', 'rb'))  # Load the precomputed similarity matrix

# Streamlit app title
st.title("Movie Recommender System")

# Dropdown to select a movie from the dataset
selected_movie_name = st.selectbox(
    'Up for some Movies',  # Dropdown label
    movies['title'].values  # List of movie titles to choose from
)

# Button that triggers the recommendation process
if st.button('Recommend'):
    # Get recommended movie names and posters based on the selected movie
    names, posters = recommend(selected_movie_name)

    # Create 5 columns to display 5 recommended movies side by side
    col1, col2, col3, col4, col5 = st.columns(5)

    # Display each recommended movie in its respective column
    with col1:
        st.text(names[0])  # Display the movie title as simple text
        st.image(posters[0])  # Display the movie poster
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])
