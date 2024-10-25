import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Load_Data:

    def __init__(self):
        self.data_file = 'data_movies_series.csv'
        self.data = None

    def check_data(self):
        if os.path.isfile(self.data_file):
            self.load_data()
            return self.data
        else:
            self.create_data() 
            if self.data is not None and not self.data.empty:
                self.clean_data()
                self.save_data()
                num_rows = self.data.shape[0]
                print(f'{num_rows} titles loaded successfully.')
                return self.data
            else:
                print("Error: No data was created. Please check the dataset files.")
                return None 

    def clean_text(self, text):
 
        if isinstance(text, str):
            cleaned = re.sub(r'[^\x00-\x7F]+', '', text)
            cleaned = cleaned.replace('#', '')
            cleaned = cleaned.replace('"', '')
            return cleaned.strip()
        return '' 

    def create_data(self):
        print(f'Starting to read data ...')

        df_netflix = None
        df_amazon = None
        df_disney = None
        df_imdb = None 
        loaded_datasets = []

        try:
            df_netflix = pd.read_csv('dataset/data_netflix.csv')
            df_netflix['stream'] = 'Netflix'
            df_netflix = df_netflix.drop(columns=['show_id', 'date_added', 'duration', 'rating'], errors='ignore')
            df_netflix = df_netflix.rename(columns={'listed_in': 'genres'})
            loaded_datasets.append('Netflix')
        except FileNotFoundError:
            print("Warning: 'data_netflix.csv' not found. Skipping this dataset.")

        try:
            df_amazon = pd.read_csv('dataset/data_amazon.csv')
            df_amazon['stream'] = 'Amazon'
            df_amazon = df_amazon.drop(columns=['show_id', 'date_added', 'duration', 'rating'], errors='ignore')
            df_amazon = df_amazon.rename(columns={'listed_in': 'genres'})
            loaded_datasets.append('Amazon')
        except FileNotFoundError:
            print("Warning: 'data_amazon.csv' not found. Skipping this dataset.")
        
        try:
            df_disney = pd.read_csv('dataset/data_disney.csv')
            df_disney['stream'] = 'Disney'
            df_disney = df_disney.drop(columns=['show_id', 'date_added', 'duration', 'rating'], errors='ignore')
            df_disney = df_disney.rename(columns={'listed_in': 'genres'})
            loaded_datasets.append('Disney')
        except FileNotFoundError:
            print("Warning: 'data_disney.csv' not found. Skipping this dataset.")

        try:
            df_imdb = pd.read_csv('dataset/data_imdb.csv')
            df_imdb['stream'] = 'Unknown'
            df_imdb = df_imdb.rename(columns={'releaseYear': 'release_year'})
            df_imdb = df_imdb.drop(columns=['numVotes', 'id','avaverageRating'], errors='ignore')
            loaded_datasets.append('IMDB')
        except FileNotFoundError:
            print("Warning: 'data_imdb.csv' not found. Skipping this dataset.")

        dataframes = [df for df in [df_imdb, df_netflix, df_amazon, df_disney] if df is not None]
        if not dataframes:
            print("Error: No datasets loaded. Cannot create combined data.")
            return

        df_all = pd.concat(dataframes, ignore_index=True, sort=False)
        df_all = df_all.infer_objects(copy=False)
        self.data = df_all

        print(f'Data from {", ".join(loaded_datasets)} loaded successfully.')
    
    def clean_data(self):
        string_columns = self.data.select_dtypes(include=['object'])
        self.data[string_columns.columns] = string_columns.apply(lambda col: col.map(self.clean_text, na_action='ignore'))
        self.data = self.data[~self.data['title'].str.strip().isin(['', ':'])]
        print(f'Data cleaned successfully.')

    def save_data(self):
        self.data.to_csv(self.data_file, index=False)
        print(f'Data saved to {self.data_file} successfully.')

    def load_data(self):
        self.data = pd.read_csv(self.data_file)
        num_rows = self.data.shape[0]
        print(f'{num_rows} titles loaded successfully.')


class Search:

    def __init__(self, data):
        self.data = data
        self.preproccess()

    def preproccess(self):
        self.description_vectorizer = TfidfVectorizer(stop_words='english')
        self.description_matrix = self.description_vectorizer.fit_transform(self.data['description'].fillna(''))
        
        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        genres_type_matrix = self.onehot_encoder.fit_transform(self.data[['genres', 'type']].fillna(''))

        self.feature_matrix = np.hstack([
            self.description_matrix.toarray(),
            genres_type_matrix,
            self.data[['release_year']].fillna(0).to_numpy()
        ])

    def search(self, query, top_n=20):
        query_vec = self.description_vectorizer.transform([query])
        
        if hasattr(query_vec, "toarray"):
            query_vec = query_vec.toarray()
        
        similarity = cosine_similarity(query_vec, self.description_matrix).flatten()

        top_indices = similarity.argsort()[-top_n:][::-1]
        return self.data.iloc[top_indices][['title', 'genres', 'type', 'release_year', 'stream','description']]


def main():

    data_loader = Load_Data()
    data = data_loader.check_data()

    if data is not None and not data.empty:  

        user_input = input("Which Movie or TV-Serie do you prefer: ")
        search_data = Search(data)
        results = search_data.search(user_input)
        print(results)

    else:
        print("No data available to search.")

if __name__ == "__main__":
    main()