from read_data import LoadData
from trainmodel import TrainModel
from recommendations import RecommendationLoader


############################## Main ############################################

def main():
    
    # Load data from CSV file
    data_loader = LoadData()
    title_data = data_loader.load_data()

    # Train model
    model = TrainModel(title_data)
    model.train()

    # Run recommendation loader
    recommendations = RecommendationLoader(model, title_data)
    recommendations.run()

if __name__ == "__main__":
    main()
