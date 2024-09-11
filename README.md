# Movie Recommendation System

This code performs data processing, feature extraction, and similarity-based movie recommendation. Below is a breakdown of its key parts:

## 1. Importing Libraries
```python
import numpy as np
import pandas as pd
import nltk
```
- **`numpy`** is used for numerical operations.
- **`pandas`** is used for handling data in a DataFrame.
- **`nltk`** (Natural Language Toolkit) is used for text processing.

## 2. Reading Data
```python
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
```
Two datasets (`movies` and `credits`) are loaded using `pandas`. These contain movie details and corresponding credits (like cast and crew).

## 3. Merging Datasets
```python
movies = movies.merge(credits, on='title')
```
The two DataFrames are merged on the `title` column so that each movie has both its details and credits combined.

## 4. Selecting Relevant Columns
```python
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
```
A subset of columns from the `movies` DataFrame is selected for further processing.

## 5. Handling Missing Data
```python
movies.isnull().sum()
movies.dropna(inplace=True)
```
Missing values are checked and removed to ensure the data is clean.

## 6. Removing Duplicates
```python
movies.duplicated().sum()
```
Duplicated records are checked (though none are removed in this step).

## 7. Processing Genres, Keywords, Cast, Crew
- The `genres`, `keywords`, `cast`, and `crew` columns are in JSON-like format, so the **`ast.literal_eval`** function is used to parse them.
   
- **Genres and Keywords**:
```python
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
```
Each genre and keyword is extracted and stored as a list of names.

- **Cast (Top 3 Actors)**:
```python
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert3)
```
The cast list is limited to the top 3 actors for each movie.

- **Director**:
```python
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)
```
The director's name is extracted from the `crew` column.

## 8. Text Preprocessing
```python
movies['overview'] = movies['overview'].apply(lambda x: x.split())
```
The `overview` text is split into individual words.

```python
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
```
A new `tags` column is created by concatenating all relevant textual features (overview, genres, keywords, cast, and crew) into one list of words for each movie.

```python
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
```
The `tags` are combined into a single string of space-separated words and converted to lowercase for normalization.

## 9. Stemming (Reducing Words to their Base Form)
```python
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)
```
A **Porter Stemmer** from NLTK is used to reduce words to their root form (e.g., "loving" becomes "love").

## 10. Creating Word Vectors
```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
```
A **CountVectorizer** is used to convert the `tags` into a matrix of word counts (word vectors). The vocabulary is limited to the 5000 most common words, and English stop words are excluded.

## 11. Cosine Similarity
```python
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
```
Cosine similarity is computed between movie vectors to quantify the similarity between different movies.

## 12. Movie Recommendation Function
```python
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(similarity[movie_index])), reverse=True, key=lambda x: x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
```
This function takes a movie title as input and recommends the top 5 similar movies based on cosine similarity.

## 13. Saving the Model
```python
import pickle
pickle.dump(new_df, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
```
The processed DataFrame (`new_df`) and similarity matrix are saved as pickle files for later use.

## Example of Use
```python
recommend('Avatar')
```
This will output the top 5 recommended movies similar to *Avatar*.

---

This code builds a movie recommendation system by:
- Preprocessing movie metadata (genres, keywords, cast, crew, etc.),
- Creating word vectors from textual data,
- Computing cosine similarity between movie vectors,
- Recommending similar movies based on the cosine similarity.
```
