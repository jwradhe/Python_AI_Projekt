# Supervised Learning - TV-Show recommender


## How to run program

**Before running program**

First thing to do is to extract TMDB_tv_dataset_v3.zip in dataset folder so that it contains TMDB_tv_dataset_v3.csv.

**Running program**

Start main.py and it will load dataset and ask for a title to get recommendations from, also how many recommendations wanted. Then enter and you will have those recommendations presented on screen.



## Specification

**TV-Show recommender**

This program will recommend you what tv-show to view based on what you like.
You will tell what tv-show you like and how many recommendations wanted, then you will get that 
amount of recommendations of tv-shows in order of rank from your search.

### Data Source:
I will use a dataset from TMBD

https://www.kaggle.com/datasets/asaniczka/full-tmdb-tv-shows-dataset-2023-150k-shows

### Model:
I must first preprocess data with vectorization so that i can train it in NearestNeighbors (NN) alhorithm with cosine distance. Later use NearestNeighbors (NN) in combination with K-NearestNeighbors (K-NN) alhorithm.

### Features:
1.  Load data from dataset and preprocessing.
2.  Model training with NN & k-NN algorithm.
3.  User input
4.  Recommendations

### Requirements:
1. Title data:
    * Title
    * Genres
    * First/last air date
    * Vote count/average
    * Director
    * Description
    * Networks
    * Spoken languages
    * Number of seasons/episodes
2. User data:
    * What Movie / TV-Show prefers
    * Number of recommendations wanted

### Libraries
  * pandas: Data manipulation and analysis
  * scikit-learn: machine learning algorithms and preprocessing
  * scipy: A scientific computing package for Python
  * time: provides various functions for working with time
  * os: functions for interacting with the operating system
  * re: provides regular expression support
  * textwrap: Text wrapping and filling
    
### Classes
  1. LoadData
     * load_data
     * read_data
     * clean_data
  2. ImportData
     * load_dataset
     * create_data
     * clean_data
     * save_data
  3. TrainModel
     * train
     * recommend
     * preprocess_title_data
     * preprocess_target_data
  4. UserData
     * input
     * n_recommendations
  5. RecommendationLoader
     * run 
     * get_recommendations
     * display_recommendations
     * get_explanation
     * check_genre_overlap
     * check_created_by_overlap
     * extract_years
     * filter_genres

### References   
   * https://scikit-learn.org/dev/modules/generated/sklearn.neighbors.NearestNeighbors.html
   * https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
   * https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.StandardScaler.html
   * https://scikit-learn.org/0.16/modules/generated/sklearn.decomposition.TruncatedSVD.html
   * https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.hstack.html
   * https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html




