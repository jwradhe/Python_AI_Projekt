import pandas as pd
from import_data import ImportData


#########################################################################
#### Class: LoadData                                                
#########################################################################
class LoadData:
    def __init__(self):
        self.data = None
        self.filename = 'TMDB_tv_dataset_v3.csv'


    ###########################################################
    #### Function: load_data                               
    ###########################################################
    def load_data(self):
        self.read_data()
        self.clean_data()
        print(f'{self.data.shape[0]} titles loaded successfully.')
        return self.data


    ###########################################################
    #### Function: read_data                             
    ###########################################################
    def read_data(self):
        print("Starting to read data ...")
        try:
            # Try to Read CSV file
            self.data = pd.read_csv('data.csv')
            print(f'{self.data.shape[0]} rows read successfully.')
        except FileNotFoundError:
            print("No data.csv file found. Attempting to import data...")
            # If CSV file not found, try to import data from datasets instead
            try:
                data_importer = ImportData()
                data_importer.create_data(self.filename)
                data_importer.clean_data()
                data_importer.save_data()
                self.data = pd.read_csv('data.csv')
                print(f'{self.data.shape[0]} rows imported successfully.')
            except Exception as e:
                print(f"Error during data import process: {e}")


    ###########################################################
    #### Function: clean_data                               
    ###########################################################
    def clean_data(self):
        # Function to split a string into a list, or use an empty list if no valid data
        def split_to_list(value):
            if isinstance(value, str):
                # Strip and split the string, and remove any empty items
                return [item.strip() for item in value.split(',') if item.strip()] 
            return []
        
        data_start = self.data.shape[0]
        
        # Split genres, spoken_languages, networks, and created_by
        self.data['genres'] = self.data['genres'].apply(split_to_list)
        self.data['spoken_languages'] = self.data['spoken_languages'].apply(split_to_list)
        self.data['networks'] = self.data['networks'].apply(split_to_list)
        self.data['created_by'] = self.data['created_by'].apply(split_to_list)

        # Drop rows that are not in English
        self.data = self.data[self.data['original_language'] == 'en']

        # Drop rows with empty lists in genres or spoken_languages
        self.data = self.data[
            self.data['genres'].map(lambda x: len(x) > 0) &
            self.data['spoken_languages'].map(lambda x: len(x) > 0) &
            self.data['networks'].map(lambda x: len(x) > 0) 
        ]

        # Count rows that were dropped
        rows_dropped = data_start - len(self.data)

        print('Data cleaned successfully, dropped ' + str(rows_dropped) + ' rows.')
