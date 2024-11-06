from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack, csr_matrix
import time

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')


############################## Train model ##############################
class TrainModel:
    def __init__(self, title_data):
        self.title_data = title_data

        # Settings for vectorization
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=0.01, max_df=0.5)

        # Settings for nearest neighbors
        self.model = NearestNeighbors(metric='cosine')  
        self.scaler = StandardScaler()

        # Settings for SVD
        self.svd = TruncatedSVD(n_components=300) 


# ---------------------- Function: train ----------------------
    def train(self):
        print("Starting to train model ...")

        start = time.time()

        # Preprocess title data
        preproccessed_data = self.preprocess_title_data()

        # Train the NearestNeighbors model
        self.model.fit(preproccessed_data)

        stop = time.time()

        # Count time for training
        elapsed_time = stop - start

        print(f'Trained model successfully in {elapsed_time:.2f} seconds.')


# ------------------------ Function: recommend ------------------------
    def recommend(self, target_row, num_recommendations=40):    

        # Preprocess target data
        target_vector = self.preprocess_target_data(target_row)

        # Use NearestNeighbors model as input to K-nearest neighbors
        distances, indices = self.model.kneighbors(target_vector, n_neighbors=num_recommendations)
        recommendations = self.title_data.iloc[indices[0]].copy()
        recommendations['distance'] = distances[0]

        # Filter recommendations
        recommendations = recommendations[
            (recommendations['name'].str.lower() != target_row['name'].lower()) & 
            (recommendations['distance'] < 0.5) 
        ]
        return recommendations.head(num_recommendations)


# ---------------------- Function: preprocess_data ----------------------
    def preprocess_title_data(self):
        # Combine text fields in a new column for vectorization
        self.title_data['combined_text'] = (
            self.title_data['overview'].fillna('').apply(str) + ' ' +
            self.title_data['genres'].fillna('').apply(str) + ' ' +
            self.title_data['created_by'].fillna('').apply(str)
        )

        # Process combined_text column with vectorizer
        text_features = self.vectorizer.fit_transform(self.title_data['combined_text'])
        text_features = self.svd.fit_transform(text_features)
        
        # Scale numerical features in the DataFrame using a scaler
        self.numerical_data = self.title_data.select_dtypes(include=['number'])

        # Include ratings in numerical features
        if 'vote_average' in self.numerical_data.columns:
            self.numerical_data = self.numerical_data[['vote_average']]

        # Scale numerical features
        numerical_features = self.scaler.fit_transform(self.numerical_data)
        numerical_features_sparse = csr_matrix(numerical_features) 
    
        # Combine text and numerical features
        combined_features = hstack([csr_matrix(text_features), numerical_features_sparse])

        return combined_features
    

# ---------------------- Function: preprocess_target_data ----------------------   
    def preprocess_target_data(self, target_row):
        # Create feature vector for target row
        target_text_vector = self.vectorizer.transform([target_row['combined_text']])
        target_text_vector = self.svd.transform(target_text_vector)
        
        # Process numerical features of the referens target
        target_numerical = target_row[self.numerical_data.columns].values.reshape(1, -1)
        target_vector = hstack([csr_matrix(target_text_vector), csr_matrix(self.scaler.transform(target_numerical))])

        return target_vector