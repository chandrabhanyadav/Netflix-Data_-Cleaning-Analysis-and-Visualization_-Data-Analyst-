# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('netflix.csv')

# Display the first few rows of the dataset
print(data.head())
print(data.describe())
print(data.info())

# Checking for missing values
print("Missing values per column:\n", data.isnull().sum())

# Checking for duplicates
print("Number of duplicate rows:", data.duplicated().sum())

# Dropping duplicates if any
data.drop_duplicates(inplace=True)

# Handling missing values
data = data.assign(
    director=data['director'].fillna('Not Given'),
    country=data['country'].fillna('Unknown')
)

# Convert 'date_added' to datetime format
data['date_added'] = pd.to_datetime(data['date_added'], errors='coerce')

# Checking data types after conversion
print("Data types after conversion:\n", data.dtypes)

# Exploratory Data Analysis (EDA)

# 1. Content Type Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='type', data=data, hue='type', palette='Set2')
plt.title('Distribution of Content by Type')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()

# 2. Most Common Genres
data['genres'] = data['listed_in'].apply(lambda x: x.split(', '))
all_genres = sum(data['genres'], [])
genre_counts = pd.Series(all_genres).value_counts().head(10)

plt.figure(figsize=(8, 6))
sns.barplot(x=genre_counts.values, y=genre_counts.index, hue=genre_counts.index, dodge=False, palette='Set3')
plt.title('Most Common Genres on Netflix')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()

# 3. Content Added Over Time
data['year_added'] = data['date_added'].dt.year
data['month_added'] = data['date_added'].dt.month

plt.figure(figsize=(12, 6))
sns.countplot(x='year_added', data=data, hue='year_added', palette='coolwarm', order=sorted(data['year_added'].dropna().unique()))
plt.title('Content Added Over Time')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 4. Top 10 Directors with the Most Titles
top_directors = data[data['director'] != 'Not Given']['director'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_directors.values, y=top_directors.index, hue=top_directors.index, dodge=False, palette='Blues_d')
plt.title('Top 10 Directors with the Most Titles')
plt.xlabel('Number of Titles')
plt.ylabel('Director')
plt.show()

# 5. Rating Distribution
ratings = data['rating'].value_counts().reset_index()
ratings.columns = ['Rating', 'Count']

plt.figure(figsize=(8, 6))
sns.barplot(x='Rating', y='Count', data=ratings, hue='Rating', dodge=False, palette='viridis')
plt.title('Ratings on Netflix')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# 6. Word Cloud of Movie Titles
movie_titles = data[data['type'] == 'Movie']['title']
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(' '.join(movie_titles))

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Movie Titles')
plt.show()

# 7. Monthly Releases of Movies and TV Shows
monthly_movie_release = data[data['type'] == 'Movie']['month_added'].value_counts().sort_index()
monthly_series_release = data[data['type'] == 'TV Show']['month_added'].value_counts().sort_index()

plt.figure(figsize=(10, 5))
plt.plot(monthly_movie_release.index, monthly_movie_release.values, label='Movies', marker='o')
plt.plot(monthly_series_release.index, monthly_series_release.values, label='TV Shows', marker='o')
plt.xlabel("Months")
plt.ylabel("Frequency of Releases")
plt.title("Monthly Releases of Movies and TV Shows on Netflix")
plt.legend()
plt.grid(True)
plt.show()

# 8. Yearly Releases of Movies and TV Shows
yearly_movie_releases = data[data['type'] == 'Movie']['year_added'].value_counts().sort_index()
yearly_series_releases = data[data['type'] == 'TV Show']['year_added'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
plt.plot(yearly_movie_releases.index, yearly_movie_releases.values, label='Movies', marker='o')
plt.plot(yearly_series_releases.index, yearly_series_releases.values, label='TV Shows', marker='o')
plt.xlabel("Years")
plt.ylabel("Frequency of Releases")
plt.title("Yearly Releases of Movies and TV Shows on Netflix")
plt.legend()
plt.grid(True)
plt.show()

# ML Section: Predicting Content Type

# Step 1: Feature Engineering
# We will predict 'type' (Movie or TV Show) based on 'rating', 'year_added', and 'country'
# First, we need to encode 'type' as 0 for 'Movie' and 1 for 'TV Show'
label_encoder = LabelEncoder()
data['type_encoded'] = label_encoder.fit_transform(data['type'])

# Step 2: Selecting Features
features = data[['rating', 'year_added', 'country']].copy()

# Handle missing or NaN values
features = features.fillna('Unknown')

# Step 3: Encode categorical features ('rating', 'country') using LabelEncoder
features['rating'] = label_encoder.fit_transform(features['rating'])
features['country'] = label_encoder.fit_transform(features['country'])

# Step 4: Split the data into training and testing sets
X = features
y = data['type_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train Logistic Regression Model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the Model
print("Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Conclusion
print("Project completed. Key insights include the distribution of content types, most common genres, rating distribution, and trends over time. Simple machine learning model successfully predicted content type.")
