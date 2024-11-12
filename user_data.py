###############################################################
#### Class: UserData                                                                
###############################################################
class UserData:
    def __init__(self):
        self.user_data = {} 
        self.n_rec = 10

    ###########################################################
    #### Function: title                               
    ###########################################################
    def title(self):
        # Ask for user input
        print("#" * 100)
        title = input("\nPlease enter the title of TV-Series you prefer: ")
        self.user_data['title'] = title.strip().lower()
        return self.user_data
    
    ###########################################################
    #### Function: n_recommendations                                
    ###########################################################
    def n_recommendations(self):
        # Ask for number of recommendations
        while True:
            n_rec = input("How many recommendations do you want (minimum 5): ")
            try:
                n_rec = int(n_rec.strip())
                if n_rec < 5:
                    print("Please enter a number greater than or equal to 5: ")
                else:
                    self.user_data['n_rec'] = n_rec
                    break
            except ValueError:
                print("Please enter a valid number.")