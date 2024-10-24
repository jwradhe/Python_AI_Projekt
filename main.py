import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        # Remove non-ASCII characters, # and " from title
        cleaned = re.sub(r'[^\x00-\x7F]+', '', text)
        cleaned = cleaned.replace('#', '')
        cleaned = cleaned.replace('"', '')
        return cleaned.strip()

    def create_data(self):
        print(f'Starting to read data ...')

        df_netflix = None
        df_amazon = None
        df_disney = None
        df_imdb = None 
        loaded_datasets = []

        # Load datasets Netflix, Amazon, and Disney
        try:
            df_netflix = pd.read_csv('dataset/data_netflix.csv')
            loaded_datasets.append('Netflix')
        except FileNotFoundError:
            print("Warning: 'data_netflix.csv' not found. Skipping this dataset.")

        try:
            df_amazon = pd.read_csv('dataset/data_amazon.csv')
            loaded_datasets.append('Amazon')
        except FileNotFoundError:
            print("Warning: 'data_amazon.csv' not found. Skipping this dataset.")
        
        try:
            df_disney = pd.read_csv('dataset/data_disney.csv')
            loaded_datasets.append('Disney')
        except FileNotFoundError:
            print("Warning: 'data_disney.csv' not found. Skipping this dataset.")

        # Load IMDB dataset and rename column
        try:
            df_imdb = pd.read_csv('dataset/data_imdb.csv')
            df_imdb = df_imdb.rename(columns={'releaseYear': 'release_year'})
            loaded_datasets.append('IMDB')
        except FileNotFoundError:
            print("Warning: 'data_imdb.csv' not found. Skipping this dataset.")

        # Create a list to hold non-empty dataframes
        dataframes = [df for df in [df_imdb, df_netflix, df_amazon, df_disney] if df is not None]

        # Check if any dataframes were loaded
        if not dataframes:
            print("Error: No datasets loaded. Cannot create combined data.")
            return

        # Concatenate all datasets
        df_all = pd.concat([df_imdb, df_netflix, df_amazon, df_disney], ignore_index=True, sort=False)

        # Forward-fill and backward-fill the entire dataframe
        df_all.ffill(inplace=True)
        df_all.bfill(inplace=True)

        df = df_all.groupby(['title', 'release_year'], as_index=False).first()
        df = df.infer_objects(copy=False)
        self.data = df

        print(f'Data from {", ".join(loaded_datasets)} loaded successfully.')
    
    def clean_data(self):
        # Clean the dataset
        string_columns = self.data.select_dtypes(include=['object'])
        self.data[string_columns.columns] = string_columns.apply(lambda col: col.map(self.clean_text))
        self.data = self.data[~self.data['title'].str.strip().isin(['', ':'])]
        print(f'Data cleaned successfully.')

    def save_data(self):
        # Save cleaned data to CSV
        self.data.to_csv(self.data_file, index=False)
        print(f'Data saved to {self.data_file} successfully.')

    def load_data(self):
        # Load data from CSV
        self.data = pd.read_csv(self.data_file)
        num_rows = self.data.shape[0]
        print(f'{num_rows} titles loaded successfully.')


def main():

    data_loader = Load_Data()
    data = data_loader.check_data()

    if data is not None and not data.empty:  
        user_input = input("Which Movie or TV-Series do you prefer?: ")

    else:
        print("No data available to search.")

if __name__ == "__main__":
    main()