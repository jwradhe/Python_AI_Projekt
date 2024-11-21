import re
import os
import pandas as pd


###############################################################
#### Class: ImportData                                                              
###############################################################
class ImportData:

    def __init__(self):
        self.data = None
        self.loaded_datasets = []


    ###########################################################
    #### Function: load_dataset                              
    ###########################################################
    def load_dataset(self, dataset_path):
        # Load data from dataset CSV file
        try:
            df = pd.read_csv(os.path.join(f'dataset', dataset_path))
            return df
        except FileNotFoundError:
            print(f'Warning: "{dataset_path}" not found. Skipping this dataset.')
            return None


    ###########################################################
    #### Function: create_data                              
    ###########################################################
    def create_data(self, filename):
        try:
            self.data = self.load_dataset(filename)
            print(f'Imported data successfully.')
        except FileNotFoundError:
            print("No data imported, missing dataset")
            return None


    ###########################################################
    #### Function: clean_data                               
    ###########################################################
    def clean_data(self):
        if self.data is not None:
            # Drop unnecessary columns
            df_cleaned = self.data.drop(columns=['adult', 'poster_path', 'production_companies', 
            'in_production','backdrop_path','production_countries','status','episode_run_time',
            'original_name', 'popularity', 'tagline','homepage'], errors='ignore')
            
            # Clean text from non-ASCII characters
            text_columns = ['name', 'overview','spoken_languages']
            masks = [df_cleaned[col].apply(lambda x: isinstance(x, str) and bool(re.match(r'^[\x00-\x7F]*$', x)))
                     for col in text_columns]
            combined_mask = pd.concat(masks, axis=1).all(axis=1)

            self.data = df_cleaned[combined_mask]

            print(f'Data cleaned. {self.data.shape[0]} records remaining.')
        else:
            print("No data to clean. Please load the dataset first.")


    ###########################################################
    #### Function: save_data                              
    ###########################################################
    def save_data(self):
        if self.data is not None:
            try:
                # Sava dataframe to CSV
                self.data.to_csv('data.csv', index=False)
                print(f'Data saved to data.csv.')
            except Exception as e:
                print(f'Error saving data: {e}')
        else:
            print("No data to save. Please clean the data first.")