import pandas as pd
import re
import os

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
            cleaned = cleaned.replace('#', '')
            cleaned = cleaned.replace('"', '')
            return cleaned.strip()
        return '' 

    def load_dataset(self, dataset_path, stream):
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

        print(f'Data from {", ".join(self.loaded_datasets)} imported.')

    def clean_data(self):
        string_columns = self.data.select_dtypes(include=['object'])
        self.data[string_columns.columns] = string_columns.apply(lambda col: col.map(self.clean_text, na_action='ignore'))
        self.data = self.data[~self.data['title'].str.strip().isin(['', ':'])]
        print(f'Data cleaned')


class UserData:

    def __init__(self):
        self.user_data = None

    def input(self):
        self.user_data = input("Which Movie or TV-Serie do you prefer: ")
        return self.user_data.lower()


class Recommendations:

    def __init__(self):
        self.result = None

    def get_recommendations(self, user_data, title_data):
        if title_data is not None and not title_data.empty: 

            self.results = "Här ska de komma rekommendationer"

            print(self.results)
        else:
            print("No data available to search.")


def main():

    data_loader = LoadData()
    title_data = data_loader.load_data()

    user_data = UserData()
    user_input = user_data.input()

    recommendations = Recommendations()
    recommendations.get_recommendations(user_data, title_data)

if __name__ == "__main__":
    main()