# Implementing content-based filtering using neural networks


<br></br>
# Overview of this project

This project is a content-based recommendation system that utilizes neural networks to suggest movies to users. It's built on the 100k Movielens dataset, enabling the system to learn from a substantial set of user-movie interactions and preferences. The goal is to provide accurate, personalized movie recommendations by analyzing the dataset with advanced neural network techniques.

<br></br>
# Scope and Context

While many industry-leading recommendation engines utilize a combination of collaborative and content-based filtering, this project aims to focus specifically on content-based filtering. The 100k Movielens dataset is used for both training and testing of this recommendation system. Accordingly, this project seeks to understand the relationship between user and movie interactions. With that in mind, the current project aims to first build a functional recommendation system with the possibility of later upscaling both the dataset and features examined. 

For the moment, this project will focus on processing data and honing the neural networks associated with the recommendation system. 

<br></br>
# Goals
- Gather and understand the Movielens 100k dataset
- Explore the 100k Movielens dataset, and identify areas for processing
- Isolate and engineer features of interest for both users and movies
- Utilize Keras and Tensorflow to develop two neural networks representing users and movies respectively
- Iteratively test the model, identifying areas for increasing accuracy

<br></br>
# Technology Stack

Programming Language: Python 
Libraries used: Tensorflow/Keras, Pandas, Numpy, Matplotlib, sklearn, tabulate

<br></br>
# Model Architecture

Note: The model architecture is currently a work in progress. Specifics of the neural networks are TBD (i.e. # nodes, # hidden layers etc.)
<img src="https://github.com/nick-ching23/Movie_Recommender/assets/67199495/14080969-9a12-4f52-bdb3-11583188f156" width="700" height="500">



User Neural Network & Neural Network: 

The User and Movie neural networks consist of a sequence of dense layers. 
- The first layer consists of a 256-neuron dense layer with ReLU activation, followed by a 128-neuron layer, also with ReLU.
- the final layer in this network has an N number of output neurons, set to 32 in this case (to be changed)

The outputs of both networks are normalized using L2 normalization.
The dot product of these two normalized vectors is then computed. This represents the interaction between user and item features.

<br></br>
# About the dataset

The original dataset has roughly 9000 movies rated by 600 users with ratings on a scale of 0.5 to 5 in 0.5 step increments. For each movie, the dataset provides a movie title, release date, and one or more genres. For example "Toy Story 3" was released in 2010 and has several genres: "Adventure|Animation|Children|Comedy|Fantasy". This dataset contains little information about users other than their ratings.

This dataset is used to create training vectors for the neural networks described below. 

<br></br>
# Data Processing & Feature Engineering
Explain how you processed the Movielens dataset.
Describe the feature engineering steps to highlight how you identified key features of user-movie interactions.

Since Ratings per user and Genres are the only information available, we generate the following: 

movie_train: movieId,	year,	ave rating, one-hot-encoding for genres of each movie
user_train: userId,  rating count,  rating ave, one-hot-encoding containing average rating per genre 

[Processing steps]
- removed movies with "(no genres listed)" and no year listed
- introduced duplicates for underrepresented movies. Found that underrepresented movies resulted in overemphasized recommendations because of the feature's distinctness

<br></br>
# Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- A Google account
- Internet access

### Installation

Follow these steps to get your development environment set up:

1. **Clone the Repository**

   Open your terminal and run the following git command:
     git clone https://github.com/nick-ching23/movie_recommender.git

2. **Unzip the Data Files**
    Navigate to the downloaded repository directory, find `data.zip`, and unzip it. This can typically be done by right-clicking and selecting "Extract All..." or using a command line:

3. **Open Google Colab**

    Visit [Google Colab](https://colab.research.google.com/) and sign in with your Google account.

4. **Set Up the Notebook**

    In Google Colab, go to `File > Open notebook`. Switch to the `GitHub` tab and enter your repository's information to find the notebook. Select it to open.

5. **Upload the Data Files**

    In the Colab notebook, find the cell that is set up for uploading files (it will have `files.upload()` in it). Run this cell and select the unzipped files (`content_item_train.csv`, `user_item_train.csv`, `y_train.csv`, `item_vecs.csv`,
    `movies.csv`) to upload them.

    <img src="https://github.com/nick-ching23/Movie_Recommender/assets/67199495/3db1a605-4670-4aeb-b994-7f72ae57f179" width="400" height="200">



7. **Run the Notebook**


<br></br>
# Results
The current model utilizes MSE (Mean Squared Error) as a cost function with an Adam Optimizer function for updating network weights during training. 

![Screenshot 2023-12-05 at 11 15 05 AM](https://github.com/nick-ching23/Movie_Recommender/assets/67199495/9c1fbb9a-761b-4968-9e42-837efd25bf6d)

The resulting model illustrates ~10.8% loss after 30 epochs of training.

![Screenshot 2023-12-05 at 11 18 43 AM](https://github.com/nick-ching23/Movie_Recommender/assets/67199495/c2b0ecbd-1f40-4ffa-a35e-e17acb9d044d)

The testing set shows an 11.48% loss which is a comparable loss rate. This illustrates overfitting is not a significant issue. 


Future development will focus on increasing accuracy of the model through testing variations in data processing as well as the hyperparameters for the above network architecture

