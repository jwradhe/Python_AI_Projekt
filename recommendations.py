from user_data import UserData
import pandas as pd
import textwrap


###############################################################
#### Class: RecommendationLoader                                
###############################################################
class RecommendationLoader:
    def __init__(self, model, title_data):
        self.model = model
        self.title_data = title_data


    ###########################################################
    #### Function: run                                 
    ###########################################################
    def run(self):
        while True: 
            user_data = UserData()
            user_data.title() 
            user_data.n_recommendations() 

            # Exit the program if writing exit or quit.
            if user_data.user_data['title'] in ['exit', 'quit']:
                print("Program will exit now. Thanks for using!")
                break
            
            # Find a row in dataset to use as referens.
            target_row = self.title_data[self.title_data['name'].str.lower() == user_data.user_data['title']]

            # If no match found, loop and try again.
            if target_row.empty:
                print(f"No match found for '{user_data.user_data['title']}'. Try again.")
                continue
            
            # If match found, get recommendations.
            target_row = target_row.iloc[0]
            self.get_recommendations(target_row, user_data.user_data)
            print("#" * 100)
            print("\nWrite 'exit' or 'quit' to end the program.")

    
    ###########################################################
    #### Function: get_recommendations                                
    ###########################################################
    def get_recommendations(self, target_row, user_data):
        recommendations = pd.DataFrame()
        n_recommendations = user_data['n_rec']

        # Get more recommendations and filter untill n_recommendations is reached
        while len(recommendations) < n_recommendations:
            additional_recommendations = self.model.recommend(target_row, num_recommendations=20)     
            additional_recommendations = additional_recommendations[~additional_recommendations.index.isin(recommendations.index)] 
            additional_recommendations = self.filter_genres(additional_recommendations, target_row)
            recommendations = pd.concat([recommendations, additional_recommendations])

        # Make sure we give n_recommendations recommendations
        recommendations = recommendations.head(n_recommendations)    

        self.display_recommendations(user_data, recommendations, n_recommendations, target_row)


    ###########################################################
    #### Function: display_recommendations                                
    ###########################################################
    def display_recommendations(self, user_data, recommendations, n_recommendations, target_row):
        print(f'\n{n_recommendations} recommendations based on "{user_data["title"]}":\n')

        # Width on printed recommendations
        width = 100

        # Print recommendations if there are any
        if not recommendations.empty:
            # print(f"{'Title':<40} {'Genres':<60} {'Networks':<30}")
            print("#" * width)

            for index, row in recommendations.iterrows():
                title = row['name']
                genres = ', '.join(row['genres']) if isinstance(row['genres'], list) else row['genres'] 
                networks = ', '.join(row['networks']) if isinstance(row['networks'], list) and row['networks'] else 'N/A'
                created_by = ', '.join(row['created_by']) if isinstance(row['created_by'], list) and row['created_by'] else 'N/A'
                rating = row['vote_average']
                vote_count = row['vote_count']
                seasons = row['number_of_seasons'] if isinstance(row['number_of_seasons'], int) else 'N/A'
                episodes = row['number_of_episodes'] if isinstance(row['number_of_episodes'], int) else 'N/A'
                overview = textwrap.fill(row["overview"], width=width)

                # Extract years fir first_air_date and last_air_date      
                first_year = self.extract_years(row["first_air_date"])
                last_year = self.extract_years(row["last_air_date"])

                # Construct title with the year range
                title_raw = f"{title} ({first_year}-{last_year})"
                title = textwrap.fill(title_raw, width=width)

                # Print recommendation
                print(f"\nTitle:    {title}")
                print(f"Genres:   {genres}")
                if not created_by == 'N/A':
                    print(f"Director: {created_by}")
                if not networks == 'N/A':
                    print(f'Networks: {networks}')
                print(f"Rating:   {rating:.1f} ({vote_count:.0f} votes)")
                if not seasons == 'N/A' and not episodes == 'N/A':
                    print(f"Seasons:  {seasons} ({episodes} episodes)")
                print(f'\n{overview}\n')
                
                # Get explanation for recommendation
                explanation = self.get_explanation(row, target_row)
                print(f"{explanation}\n")

                print("-" * width)

            print("\nEnd of recommendations.")
        else:
            print("No recommendations found.")


    ###########################################################
    #### Function: get_explanation                                 
    ###########################################################
    def get_explanation(self, row, target_row):
        explanation = []
        title = row['name']

        explanation.append(f"The title '{title}' was recommended because: \n")

        # Explain genre overlap
        genre_overlap = self.check_genre_overlap(target_row, row)
        if genre_overlap:
            overlapping_genres = ', '.join(genre_overlap)
            explanation.append(f"It shares the following genres with your preferences: {overlapping_genres}.\n")   
        
        # Explain created_by overlap
        created_by_overlap = self.check_created_by_overlap(target_row, row)
        if created_by_overlap:
            overlapping_created_by = ', '.join(created_by_overlap)
            explanation.append(f"It shares the following director with your preferences: {overlapping_created_by}.\n")

        # Explain the distance metric
        explanation.append(f"The distance metric of {round(row['distance'], 2)} indicates that it is quite similar to your preferences.")
        return ' '.join(explanation)


    ###########################################################
    #### Function: check_genre_overlap                              
    ###########################################################
    def check_genre_overlap(self, target_row, row):
        # Get genres from the target row
        target_genres = set(genre.lower() for genre in target_row['genres'])
        # Get genres from the recommended row
        recommended_genres = set(genre.lower() for genre in row['genres'])

        # Find the intersection of the target genres and recommended genres
        overlap = target_genres.intersection(recommended_genres) 

        return overlap


    ###########################################################
    #### Function: check_created_by_overlap                                
    ###########################################################
    def check_created_by_overlap(self, target_row, row):
        # Get created_by from the target row
        target_creators = set(creator.lower() for creator in target_row['created_by'])
        # Get created_by from the recommended row
        recommended_creators = set(creator.lower() for creator in row['created_by'])

        # Find the intersection of the target creators and recommended creators
        overlap = target_creators.intersection(recommended_creators)

        return overlap


    ###########################################################
    #### Function: extract_years                                 
    ###########################################################
    def extract_years(self, air_date):
        # Make sure air_date is not null
        if pd.isna(air_date):
            return "Unknown"
        # Convert float to int if needed
        if isinstance(air_date, float):
            return str(int(air_date))
        return air_date.split('-')[0]  


    ###########################################################
    #### Function: filter_genres                              
    ###########################################################
    def filter_genres(self, recommendations, target_row):
        # Get genres from the target row
        reference_genres = [genre.lower() for genre in target_row['genres']]

        # Check if the reference includes specific genres
        is_kids_reference = 'kids' in reference_genres
        is_animated_reference = 'animation' in reference_genres
        is_reality_reference = 'reality' in reference_genres
        is_documentary_reference = 'documentary' in reference_genres

        # Filter recommendations based on genre preferences
        if not is_kids_reference:
            recommendations = recommendations[~recommendations['genres'].apply(lambda x: 'kids' in [g.lower() for g in x])]
        if not is_animated_reference:
            recommendations = recommendations[~recommendations['genres'].apply(lambda x: 'animation' in [g.lower() for g in x])]
        if not is_reality_reference:
            recommendations = recommendations[~recommendations['genres'].apply(lambda x: 'reality' in [g.lower() for g in x])]
        if not is_documentary_reference:
            recommendations = recommendations[~recommendations['genres'].apply(lambda x: 'documentary' in [g.lower() for g in x])]

        return recommendations
