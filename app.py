from flask import Flask, render_template, request
from readdata import LoadData
from recommendations import RecommendationLoader
from training import TrainModel

app = Flask(__name__)

data_loader = LoadData()
title_data = data_loader.load_data()

model = TrainModel(title_data)
model.train()

recommender = RecommendationLoader(model, title_data)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():

    # Get user input
    title = request.form.get('title').strip()
    n_recommendations = int(request.form.get('n_recommendations', 10))

    # Validate user input
    if not title:
        return render_template('index.html', message="Please enter a valid TV show title.")
    
    try:
        n_recommendations = int(n_recommendations)
        if n_recommendations < 1 or n_recommendations > 50:
            raise ValueError("Number of recommendations must be between 1 and 50.")
    except ValueError as e:
        return render_template('index.html', message=str(e))

    # Get recommendations from the model
    target_row = title_data[title_data['name'].str.lower() == title.lower()]
    
    # Check if a match was found
    if target_row.empty:
        return render_template('index.html', message=f"No match found for '{title}'. Try again.")

    # Get recommendations
    target_row = target_row.iloc[0]
    user_data = {'title': title, 'n_rec': n_recommendations}
    recommendations = recommender.get_recommendations("flask", target_row, user_data)
    
    # Check if recommendations were found
    if recommendations is None or recommendations.empty:
        return render_template('index.html', message=f"Sorry, no recommendations available for {title}.")
    
    # Prepare data for display on the webpage
    recommendations_data = []

    for _, row in recommendations.iterrows():

        # Extract the first and last air dates
        first_air_date = recommender.extract_years(row['first_air_date'])
        last_air_date = recommender.extract_years(row['last_air_date'])
        if last_air_date != "Ongoing" and last_air_date:
            years = f"{first_air_date} - {last_air_date}"
        else:
            years = f"{first_air_date}"

        recommendations_data.append({
            'title': row['name'],
            'genres': ', '.join(row['genres']) if isinstance(row['genres'], list) else row['genres'],
            'overview': row['overview'],
            'rating': row['vote_average'],
            'seasons': row['number_of_seasons'],
            'episodes': row['number_of_episodes'],
            'networks': ', '.join(row['networks']) if isinstance(row['networks'], list) and row['networks'] else 'N/A',
            'years': years,
        })

    return render_template('index.html', recommendations=recommendations_data, original_title=title)


if __name__ == '__main__':
    app.run(debug=True)
