import numpy as np
import pandas as pd
import warnings

#it will ignore all the warnings which occur during worktime-------------------------------->>>>>>>>>

warnings.filterwarnings('ignore')

# Get data Set----------------------------------------------------------------------------->>>>>>>>
# Name the columns in your dataset---------------------------------------------------------->>>>>>>>

columns_name = ["user-id", "item-id", "rating", "timeStamp"]
df = pd.read_csv("ml-100k/u.data", sep='\t', names=columns_name)


movies_title = pd.read_csv("ml-100k/u.item", sep="\|", header=None)
movies_title = movies_title[[0,1]]
movies_title.columns = ["item-id", "title"]
print(movies_title.head())

# merge two data frames(df, movies_title)---------------------------------------------------->>>>>>>>>
#(on) merge the frames using a unique key which present in both dataframes------------------->>>>>>>>>

df = pd.merge(df, movies_title, on="item-id")
print(df.head())
print(df.tail())

# Exploratory Data Analysis---------------------------------------------------------->>>>>>>>>>>>>>>>>
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

# we group our dataframe and find the average rating of each and every movie------------------>>>>>>>>

# df = df.groupby('title').mean()['rating'].sort_values(ascending = False)
# print(df.head())

# df = df.groupby('title').count()['rating'].sort_values(ascending = False)
# print(df.head())


ratings = pd.DataFrame(df.groupby('title').mean()['rating'])
print(ratings.head())

# how many no of ratings these movies got(num of ratings)-------------------------------->>>>>>>>>>>>>

ratings['num of r_views'] = pd.DataFrame(df.groupby('title').count()['rating'])
# print(ratings.head())
print(ratings.sort_values(by='rating', ascending=False))

#let's create the histogram graph of the (num of ratings = no of views) each movie got using matplotlib and seaborn

plt.figure(figsize=(10, 6))
plt.hist(ratings['num of r_views'], bins = 70)
plt.show()

# let's also plot the histogram of ratings by user--------------------------------------->>>>>>>>>>>>>

plt.hist(ratings['rating'], bins=70)
plt.show()

plt.hist(ratings['num of r_views'], bins = 70)
plt.show()
# let's join two plots rating and num of ratings using seaborn--------------------------->>>>>>>>>>>>>

# sns.jointplot(x='rating',y='num of r_views', data=ratings, alpha=0.5)

# Creating Movie Recommendation-------------------------------------------------------->>>>>>>>>>>>>>

movie_matrix = df.pivot_table(index="user-id", columns="title", values="rating")
# print(movie_matrix)
ratings = ratings.sort_values('num of r_views', ascending=False)
# print(ratings.head())

# print(movie_matrix['Star Wars (1977)'])
stars_wars_user_ratings = movie_matrix['Star Wars (1977)']
# print(stars_wars_user_ratings)

# let's co-relate your stars wars movie with other movies

corrwith_other_movies = movie_matrix.corrwith(stars_wars_user_ratings)
# print(corrwith_other_movies)

corrof_starWars = pd.DataFrame(stars_wars_user_ratings, columns=['correlation'])
# print(corrof_starWars)



# let's analysis how many times a movie has been watched and how mauch is their ratings------->>>>>>>>

# df = df.groupby('title').count()['rating'].sort_values(ascending = False)
# print(df.head())


# print(df["user-id"]) #show all the user-id
# print(df["user-id"].nunique()) #gives all the unique user-id's
# print(df["item-id"].nunique()) #gives all the unique item-id's/movies



