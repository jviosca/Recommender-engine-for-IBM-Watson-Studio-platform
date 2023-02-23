# Recommender Engine for IBM Watson Studio platform
A content-based and collaborative-filtering recommendation system for 
content items and users at IBM Watson Studio platform.


### Table of Contents

1. [Project Motivation](#motivation)
2. [Results](#results)
3. [Installation](#installation)
4. [File Descriptions](#files)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Motivation<a name="motivation"></a>

The aim of this project is to build a recommender engine for content items at IBM-Watson Studio
platform using a variety of recommender approaches:
- Content-based recommendations, based on content similarity
- Collaborative-filtering, based on user similarity

The recommender presented here blends content-based and 
collaborative filtering with ranked-based 
recommendations (most popular items), offering tailored
content recommendations for different scenarios.

This project is a graded assignment from Udacity's 
[Data Scientist](https://www.udacity.com/course/data-scientist-nanodegree--nd025) 
nanodegree. The dataset was provided as part of the assignment.


## Results<a name="results"></a>

The recommender described here offers tailored recommendations for 3 different scenarios:
- When a new user enters the platform, it shows the most popular items
in the platform (items with higher number of views).
- When a user that has recently joined the platform 
(number of articles seen > 0 and < 6) enters the platform, 
the recommender shows, in addition to popular articles, articles that 
are similar to those already seen by the user.
- In the rest of the cases (i.e. when a "Old" user who has seen more than 
5 articles enters the platform), the engine shows, in addition to popular
and similar articles, articles that have been seen by similar users.

A demo of the recommender is deployed in Streamlit cloud and can be 
accessed [here](https://recommender-ibm-watson.streamlit.app/). 
The webapp showcases the recommender engine in the 3 scenarios described 
above and simulates user activity to show how recommendations are updated
accordingly.

In addition, a Jupyter notebook called "Recommendations_with_IBM.ipynb" contains 
an exploratory data analysis of the datasets and describes 
the creation of a "Recommender" class writen 
in Python that can be easily reused or repurposed. The notebook also contains
graded questions for Udacity's assignment. 

In short, the recommendation approaches in such a class use:
- User-user based collaborative filtering, to pull articles seen by similar
users ranked by descending user activity and article views. 
Similar users are computed by the dot product of the user-item matrix 
(recomputed to binary encode items that have been seen by each user) with
its transpose matrix.
- Content-based recommendations to pull similar contents 
ranked by higher to lower cosine similarity of term-frequency-inverse 
document frequency (Tf-Idf) vectorized item
text titles (i.e. items sharing a higher number of words in their titles).
- Matrix factorization and Single Value Decomposition (SVD), to bring articles
predicted to be liked by a given user from the existing user-item matrix 
given a k number of latent factors.

A longer description of the recommender approaches and dataset exploration 
can be found at the post available 
[here](https://jvros.com.es/index.php/en/on-the-roots-of-overweight-nature-or-nurture/).


## Installation <a name="installation"></a>

If you want to build your own recommender starting from this one, you can either 
fork this repository and install it on Streamlit cloud
([instructions](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app)) 
or clone it and run Streamlit [locally](https://docs.streamlit.io/library/get-started/create-an-app) 
on your computer.

Besides Streamlit, other python libraries needed are specified in the 
requirements.txt file. The code should run with no issues using 
Python versions 3.*.


## File Descriptions <a name="files"></a>

The Streamlit webapp uses the following files:
- recommender.py (builds the Recommender python class)
- recommender_functions.py (contains functions used by the Recommender class)
- app.py (contains the code for the Streamlit app)
- requirements.txt (contains libraries that need to be imported at 
Streamlit cloud)

The jupyter notebook imports or loads data from the following scripts:
- project_tests.py (contains several grading functions)
- user_item_matrix.p (contains the user-item-matrix, as described above)
- users_dot_product.csv (used to compute user similarity)  


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

You can find the Licensing for the data at the LICENSE file. 
Otherwise, feel free to use the code here as you would like! 
