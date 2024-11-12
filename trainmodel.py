from read_data import LoadData
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack, csr_matrix
import numpy as np
import pickle
import time


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')


#########################################################################
#### Class: TrainModel                                                
#########################################################################
class TrainModel:
    def __init__(self, title_data):
        self.title_data = title_data

        # Initialize Sentence-BERT model for embeddings
        self.bert_model = SentenceTransformer('all-MiniLM-L12-v2')

        # Settings for TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=0.01, max_df=0.5)

        # Settings for Nearest Neighbors
        self.nearest_neighbors = NearestNeighbors(metric='cosine') 
        self.scaler = StandardScaler()

        # Settings for SVD
        self.svd = TruncatedSVD(n_components=300) 


    ###########################################################
    #### Function: Train                                   
    ###########################################################
    def train(self):
        print("Starting to train model ...")

        start = time.time()

        # Preprocess title data with advanced embeddings included
        preprocessed_data = self.preprocess_title_data()

        # Train Nearest Neighbors on the enhanced feature set
        self.nearest_neighbors.fit(preprocessed_data)

        print(f'Trained model successfully in {time.time() - start:.2f} seconds.')


    ###########################################################
    #### Function: get_recommendations                          
    ###########################################################
    def recommend(self, target_row, num_recommendations=40):    

        # Preprocess target data
        target_vector = self.preprocess_target_data(target_row)

        # Use Nearest Neighbors to get recommendations
        distances, indices = self.nearest_neighbors.kneighbors(target_vector, n_neighbors=num_recommendations)
        recommendations = self.title_data.iloc[indices[0]].copy()
        recommendations['distance'] = distances[0]

        # Filter recommendations
        recommendations = recommendations[
            (recommendations['name'].str.lower() != target_row['name'].lower()) & 
            (recommendations['distance'] < 0.5) 
        ]
        return recommendations.head(num_recommendations)


    ###########################################################
    #### Function: preprocess_title_data                    
    ###########################################################
    def preprocess_title_data(self):
        self.title_data['combined_text'] = (
            self.title_data['overview'].fillna('').apply(str) + ' ' +
            self.title_data['genres'].fillna('').apply(str) + ' ' +
            self.title_data['created_by'].fillna('').apply(str)
        )

        # Process text data for TF-IDF + SVD
        text_features = self.vectorizer.fit_transform(self.title_data['combined_text'])
        text_features = self.svd.fit_transform(text_features)

        # Generate Sentence-BERT embeddings
        bert_embeddings = self.load_pickle('bert_embeddings.pkl', self.title_data['combined_text'])

        # Process numerical features
        self.numerical_data = self.title_data.select_dtypes(include=['number'])
        if 'vote_average' in self.numerical_data.columns:
            self.numerical_data = self.numerical_data[['vote_average']]
        numerical_features = self.scaler.fit_transform(self.numerical_data)
        numerical_features_sparse = csr_matrix(numerical_features)

        # Combine all features
        combined_features = hstack([csr_matrix(text_features), csr_matrix(bert_embeddings), numerical_features_sparse])

        return combined_features
    

    ###########################################################
    #### Function: preprocess_target_data                    
    ###########################################################
    def preprocess_target_data(self, target_row):
        # Process target text data for TF-IDF + SVD
        target_text_vector = self.vectorizer.transform([target_row['combined_text']])
        target_text_vector = self.svd.transform(target_text_vector)
        
        # Generate Sentence-BERT embedding for target
        target_bert_embedding = self.embed_text(target_row['combined_text']).reshape(1, -1)

        # Process numerical features
        target_numerical = target_row[self.numerical_data.columns].values.reshape(1, -1)
        target_numerical_scaled = self.scaler.transform(target_numerical)

        # Combine all target features
        target_vector = hstack([csr_matrix(target_text_vector), csr_matrix(target_bert_embedding), csr_matrix(target_numerical_scaled)])

        return target_vector
    

    ###########################################################
    #### Function: load_pickle                   
    ###########################################################
    def load_pickle(self, filename, title_data):
        try:
            with open(filename, 'rb') as f:
                bert_embeddings = pickle.load(f)
        except FileNotFoundError:
            bert_embeddings = np.vstack(title_data.apply(self.embed_text).values)
            with open(filename, 'wb') as f:
                pickle.dump(bert_embeddings, f)
        return bert_embeddings
    

    ###########################################################
    #### Function: embed_text                    
    ###########################################################
    def embed_text(self, text):
        # Use Sentence-BERT to create embeddings
        return self.bert_model.encode(text, convert_to_numpy=True)
    
    
