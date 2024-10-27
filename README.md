# Supervised Learning - Movie/TV-Show recommender

## Specification
Movie/TV-Show recommender

This program will recommend you what movie or th-show to view based on what Movie/TV-Show you like.
You should be able to search for recommendations from your Movie/TV-Show title, cast, director, 
release year and also Description, and get back a recommendations with a explanation on what just this 
title might suit you.

### Data Source:
I will use 4 datasets from kaggle, 3 datasets from streaming-sites Netflix, 
Amazon Prime and Disney Plus, also 1 from a IMDB dataset.

### Model:
I will use k-Nearest Neighbors (k-NN) alhorithm that can help me find other titles based on features 
like Title, Release year, Description, Cast, Director and genres.

### Features:
1.  Load data from several data-files and preprocessing.
2.  Model training with k-NN algorithm.
3.  Search with explanation

### Requirements:
1. Title data:
  * Title
  * Genres
  * Release year
  * Cast
  * Director
  * Description
2. User data:
  * What Movie / TV-Show
  * What genre
  * Director

### Libraries
  * pandas: Data manipulation and analysis
  * scikit-learn: machine learning algorithms and preprocessing
  * beatifulsoup4: web scraping (if necessary)
    
### Classes
  1. LoadData
     * check_data
     * clean_text
     * clean_data
     * load_dataset
     * create_data
     * save_data
     * load_data
  2. UserData
     * input
  3. Recommendations
     * get_recommendations 

