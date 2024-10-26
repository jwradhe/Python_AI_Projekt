import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class LoadData:

    def __init__(self):
        self.data_file = 'data_movies_series.csv'
        self.data = None
        self.loaded_datasets = []

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
  
    def clean_data(self):
        string_columns = self.data.select_dtypes(include=['object'])
        self.data[string_columns.columns] = string_columns.apply(lambda col: col.map(self.clean_text, na_action='ignore'))
        self.data = self.data[~self.data['title'].str.strip().isin(['', ':'])]
        print(f'Data cleaned successfully.')

    def load_dataset(self, dataset_path, stream):
        print(f'dataset/{dataset_path}')
        try:
            df = pd.read_csv(f'dataset/{dataset_path}')
            df['stream'] = stream
            if stream not in 'IMDB':
                df = df.drop(columns=['show_id', 'date_added', 'duration', 'rating'], errors='ignore')
                df = df.rename(columns={'listed_in': 'genres'})
            else:
                df = df.rename(columns={'releaseYear': 'release_year'})
                df = df.drop(columns=['numVotes', 'id','avaverageRating'], errors='ignore')
            self.loaded_datasets.append(stream)
            return df
        except FileNotFoundError:
            print(f'Warning: "{dataset_path}" not found. Skipping this dataset.')

    def create_data(self):
        print(f'Starting to read data ...')
        
        df_netflix = self.load_dataset('data_netflix.csv','Netflix')
        df_amazon = self.load_dataset('data_amazon.csv','Amazon')
        df_disney = self.load_dataset('data_disney.csv','Disney')
        df_imdb = self.load_dataset('data_imdb.csv','IMDB')

        dataframes = [df for df in [df_imdb, df_netflix, df_amazon, df_disney] if df is not None]
        if not dataframes:
            print("Error: No datasets loaded. Cannot create combined data.")
            return

        df_all = pd.concat(dataframes, ignore_index=True, sort=False)
        df_all = df_all.infer_objects(copy=False)
        self.data = df_all

        print(f'Data from {", ".join(self.loaded_datasets)} loaded successfully.')

    def save_data(self):
        self.data.to_csv(self.data_file, index=False)
        print(f'Data saved to {self.data_file} successfully.')

    def load_data(self):
        self.data = pd.read_csv(self.data_file)
        num_rows = self.data.shape[0]
        print(f'{num_rows} titles loaded successfully.')


class UserData:

    def __init__(self):
        self.user_data = None

    def input(self):
        self.user_data = input("Which Movie or TV-Serie do you prefer: ")
        return self.user_data.lower()


class Search:

    def __init__(self, data):
        self.data = data
        self.preprocess()

    def preprocess(self):
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


class Recommendations:

    def __init__(self):
        self.result = None

    def get_recommendations(self, user_data, title_data):
        if title_data is not None and not title_data.empty:     
            search_data = Search(title_data)
            self.results = search_data.search(user_data)
            print(self.results)
        else:
            print("No data available to search.")


def main():

    data_loader = LoadData()
    title_data = data_loader.check_data()

    user_data = UserData()
    user_input = user_data.input()

    recommendations = Recommendations()
    recommendations.get_recommendations(user_data, title_data)

if __name__ == "__main__":
    main()