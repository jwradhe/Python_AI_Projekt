# Supervised Learning - TV-Show Recommender

## Table of Contents
1. [How to Run the Program](#how-to-run-the-program)
2. [Project Overview](#project-overview)
3. [Dataset](#dataset)
4. [Model and Algorithm](#model-and-algorithm)
5. [Features](#features)
6. [Requirements](#requirements)
7. [Libraries](#libraries)
8. [Classes](#classes)
9. [References](#references)

## How to Run the Program

### Prerequisites

1. **Download and Extract the Dataset:**
   - Download the dataset from [TMDB TV Dataset](https://www.kaggle.com/datasets/asaniczka/full-tmdb-tv-shows-dataset-2023-150k-shows).
   - Extract `TMDB_tv_dataset_v3.zip` into the `dataset/` folder, so it contains the file `TMDB_tv_dataset_v3.csv`.

2. **Install Dependencies:**
   - Install the necessary libraries listed in `requirements.txt` (see below).

### Running the Program

There are two ways to run the program, depending on whether you prefer to use the web-based interface or the command-line interface (CLI).

#### Web Interface (Flask)

To run the web-based interface (Flask application):

```bash
python app.py
```
   
This will start a local web server, and you can access the app through your browser (usually at http://127.0.0.1:5000/).
The program will load the dataset, prompt you to enter a TV show title, and ask how many recommendations you want.
   
#### Command-Line Interface (Python-GUI)

To run the command-line version of the program:
   
```bash
python main.py
```

The program will work in the terminal, asking you to enter the title of a TV show you like and how many recommendations you want.

> [!NOTE]
>  The first time the program is run, it will generate **Sentence-BERT embeddings**. This can take up to 5 minutes due to the large size of the dataset.

---

## Project Overview

The **TV-Show Recommender** is a machine learning-based program that suggests TV shows to users based on their preferences. The system uses **Nearest Neighbors (NN)** and **K-Nearest Neighbors (K-NN)** algorithms with **cosine distance** to recommend TV shows. Users provide a title of a TV show they like, and the system returns personalized recommendations based on similarity to other TV shows in the dataset.

---

## Dataset

The dataset used in this project is sourced from **TMDB** (The Movie Database). It contains over 150,000 TV shows and includes information such as:

- Title of TV shows
- Genres
- First/Last air date
- Vote count and average rating
- Director/Creator information
- Overview/Description
- Networks
- Spoken languages
- Number of seasons/episodes

Download the dataset from [here](https://www.kaggle.com/datasets/asaniczka/full-tmdb-tv-shows-dataset-2023-150k-shows).

---

## Model and Algorithm

The recommender system is based on **Supervised Learning** using the **NearestNeighbors** and **K-NearestNeighbors** algorithms. Here's a breakdown of the process:

1. **Data Preprocessing:** 
   - The TV show descriptions are vectorized using **Sentence-BERT embeddings** to create dense vector representations of each show's description.
   
2. **Model Training:**
   - The **NearestNeighbors (NN)** algorithm is used with **cosine distance** to compute similarity between TV shows. The algorithm finds the most similar shows to a user-provided title.
   
3. **Recommendation Generation:**
   - The model generates a list of recommended TV shows by finding the nearest neighbors of the input title using cosine similarity.

---

## Features

1. **Data Loading & Preprocessing:** 
   - Loads the TV show data from a CSV file and preprocesses it for model training.

2. **Model Training with K-NN:**
   - Trains a K-NN model using the **NearestNeighbors** algorithm for generating recommendations.

3. **User Input for Recommendations:**
   - Accepts user input for the TV show title and the number of recommendations.

4. **TV Show Recommendations:**
   - Returns a list of recommended TV shows based on similarity to the input TV show.

---

## Requirements

### Data Requirements:
The dataset should contain the following columns for each TV show:
- **Title**
- **Genres**
- **First/Last air date**
- **Vote count/average**
- **Director**
- **Overview**
- **Networks**
- **Spoken languages**
- **Number of seasons/episodes**

### User Input Requirements:
- **TV Show Title**: The name of the TV show you like.
- **Number of Recommendations**: The number of recommendations you want to receive (default is 10).

---

## Libraries

The following libraries are required to run the program:

- **pandas**: For data manipulation and analysis.
- **scikit-learn**: For machine learning algorithms and preprocessing.
- **scipy**: For scientific computing (e.g., sparse matrices).
- **time**: For working with time-related functions.
- **os**: For interacting with the operating system.
- **re**: For regular expression support.
- **textwrap**: For text wrapping and formatting.
- **flask**: For creating the web interface.

To install the dependencies, run:

```bash
pip install -r requirements.txt
