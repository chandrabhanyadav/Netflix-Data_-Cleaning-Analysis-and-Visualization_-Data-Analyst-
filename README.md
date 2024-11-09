# Netflix-Data_-Cleaning-Analysis-and-Visualization_-Data-Analyst-
📊 Netflix Data Analysis and Classification
Project Overview
This project involves data cleaning, exploratory data analysis (EDA), visualization, and content type prediction using machine learning on a Netflix dataset. 
The analysis uncovers various insights into the distribution of Netflix content, top genres, popular directors, rating distributions, and release trends. Additionally, 
a Logistic Regression model is used to predict the content type (Movie or TV Show).
#Dataset=(https://github.com/chandrabhanyadav/Netflix-Data_-Cleaning-Analysis-and-Visualization_-Data-Analyst-/blob/main/netflix.csv)
#Project Steps
1. Data Preprocessing
Loaded the dataset and checked its structure using .head(), .info(), and .describe().
Handled missing values for director and country.
Removed duplicate rows.
Converted date_added to datetime format for analysis.
2. Exploratory Data Analysis
Performed in-depth analysis to answer the following questions:

What is the distribution of content types (Movies vs. TV Shows)?
What are the most popular genres on Netflix?
How has the addition of content changed over time?
Who are the top 10 directors with the most content?
How are Netflix's ratings distributed?
3. Data Visualization
Count Plot: Distribution of content types.
Bar Plot: Most common genres and top directors.
Line Plot: Monthly and yearly content additions.
Word Cloud: Visualizing frequent words in movie titles.
4. Machine Learning
Feature Engineering: Selected rating, year_added, and country as features for predicting the content type.
Label Encoding: Encoded categorical features (rating and country).
Model Training: Trained a Logistic Regression model.
Model Evaluation: Achieved accuracy and generated a classification report.
#Results
The dataset primarily consists of Movies (approx. 70%).
Popular genres include Dramas, Comedies, and Action & Adventure.
The majority of content is added in recent years, especially in 2019 and 2020.
The top directors on Netflix include well-known names like Alfred Hitchcock and Martin Scorsese.
The Logistic Regression model achieved a decent accuracy in predicting content type based on ratings, release year, and country.
#Conclusion
This project provided valuable insights into Netflix's content strategy, popular genres, 
and trends over the years. Additionally, we demonstrated how machine learning can be used to classify content based on key features.
