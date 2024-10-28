import pandas as pd
import re
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from textwrap import dedent

class LoadData:
    def __init__(self):
        self.data = None
        self.loaded_datasets = []

    def load_data(self):
        self.create_data()
        self.clean_data()
        num_rows = self.data.shape[0]
        print(f'{num_rows} titles loaded successfully.')
        return self.data

    def clean_text(self, text):
        if isinstance(text, str):
            cleaned = re.sub(r'[^\x00-\x7F]+', '', text)
            cleaned = cleaned.replace('#', '').replace('"', '')
            return cleaned.strip()
        return '' 

    def load_dataset(self, dataset_path, stream):
        try:
            df = pd.read_csv(f'dataset/{dataset_path}')
            df['stream'] = stream
            if stream != 'IMDB':
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

        print(f'Data from {", ".join(self.loaded_datasets)} imported.')

    def clean_data(self):
        self.data.dropna(subset=['title', 'genres', 'description'], inplace=True)
        string_columns = self.data.select_dtypes(include=['object'])
        self.data[string_columns.columns] = string_columns.apply(lambda col: col.map(self.clean_text, na_action='ignore'))
        self.data = self.data[~self.data['title'].str.strip().isin(['', ':'])]
        self.data['genres'] = self.data['genres'].str.split(', ').apply(lambda x: [genre.strip() for genre in x])
        self.data = self.data[self.data['genres'].map(lambda x: len(x) > 0)]
        print(f'Data cleaned. {self.data.shape[0]} records remaining.')


class UserData:
    def __init__(self):
        self.user_data = None

    def input(self):
        self.user_data = input("Which Movie or TV Series do you prefer: ")
        return self.user_data.strip().lower()


class TrainModel:
    def __init__(self, title_data):
        self.recommendation_model = None
        self.title_data = title_data
        self.title_vectors = None 
        self.vectorizer = TfidfVectorizer() 
        self.preprocess_data()

    def preprocess_data(self):
        self.title_data['genres'] = self.title_data['genres'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
        self.title_data['combined_text'] = (
            self.title_data['title'].fillna('') + ' ' +
            self.title_data['director'].fillna('') + ' ' +
            self.title_data['cast'].fillna('') + ' ' +
            self.title_data['genres'] + ' ' + 
            self.title_data['description'].fillna('')
        )
        self.title_data['combined_text'] = self.title_data['combined_text'].str.lower()
        self.title_data['combined_text'] = self.title_data['combined_text'].str.replace(r'[^a-z\s]', '', regex=True)
        self.title_vectors = self.vectorizer.fit_transform(self.title_data['combined_text'])

    def preprocess_user_input(self, user_input):
        user_vector = self.vectorizer.transform([user_input])
        return user_vector

    def train(self):
        self.recommendation_model = NearestNeighbors(n_neighbors=10, metric='cosine')
        self.recommendation_model.fit(self.title_vectors)


class RecommendationLoader:
    def __init__(self, model, title_data):
        self.model = model
        self.title_data = title_data

    def run(self):
        while True: 
            user_data = UserData()
            user_input = user_data.input()

            if user_input in ['exit', 'quit']:
                print("Program will exit now. Thanks for using!")
                break

            self.get_recommendations(user_input) 
            print("\nWrite 'exit' or 'quit' to end the program.")
            
    def get_recommendations(self, user_data):
        user_vector = self.model.preprocess_user_input(user_data)   
        distances, indices = self.model.recommendation_model.kneighbors(user_vector, n_neighbors=10) 
        recommendations = self.title_data.iloc[indices[0]]

        self.display_recommendations(user_data, recommendations)

    def display_recommendations(self, user_data, recommendations):
        print(f'\nRecommendations based on "{user_data}":\n')

        if not recommendations.empty:
            movie_recommendations = recommendations[recommendations['type'] == 'Movie']
            tv_show_recommendations = recommendations[recommendations['type'] == 'TV Show']

            if not movie_recommendations.empty:
                print("\n#################### Recommended Movies: ####################")
                for i, (_, row) in enumerate(movie_recommendations.iterrows(), start=1):
                    print(dedent(f"""
                        {i}. {row['title']} ({row['release_year']}) ({row['genres']})
                        Description: {row['description']}
                        Director: {row['director']}
                        Cast: {row['cast']}
                        
                        ===============================================================
                    """))

            if not tv_show_recommendations.empty:
                print("\n#################### Recommended TV Shows: ####################")
                for i, (_, row) in enumerate(tv_show_recommendations.iterrows(), start=1):
                    print(dedent(f"""
                        {i}. {row['title']} ({row['release_year']}) ({row['genres']})
                        Description: {row['description']}
                        Director: {row['director']}
                        Cast: {row['cast']}
                        
                        ===============================================================
                    """))
        else:
            print("No recommendations found.")


def main():
    data_loader = LoadData()
    title_data = data_loader.load_data()

    model = TrainModel(title_data)
    model.train()

    recommendations = RecommendationLoader(model, title_data)
    recommendations.run()

if __name__ == "__main__":
    main()
