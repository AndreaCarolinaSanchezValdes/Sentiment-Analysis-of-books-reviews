import pandas as pd
import plotly.express as px
import re
import string
from wordcloud import STOPWORDS, WordCloud
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import nltk as nltk
from sklearn.metrics import confusion_matrix
import numpy as np

# import data from csv files
books_data = pd.read_csv(r"C:\\Users\\E_jr1\\Downloads\\books_data-1.csv")
books_rating = pd.read_csv(r"C:\\Users\\E_jr1\\Downloads\\Books_rating.csv")

# show all columns
pd.set_option('display.max_columns', None)

# iterating the columns
print('-----Dataframe books_data (column name)')
for col in books_data.columns:
    print(col)

print('-----Dataframe books_data (column name)')
for col in books_rating.columns:
    print(col)


print('-----Dataframe books_data (number of rows)')
print(len(books_data))

print('-----Dataframe books_rating (number of rows)')
print(len(books_rating))


# print head of dataframes
print('-----Dataframe books_data')
print(books_data.head())

print('-----Dataframe books_rating')
print(books_rating.head())


# initial exploration of dataframes
print('-----Dataframe books_data summary')
print(books_data.describe(include="all"))

print('-----Dataframe books_rating summary')
print(books_rating.describe(include="all"))


# Sanity checks

# Treating duplicates

print('-----Treating duplicates')


def describe_duplicates(df: pd.DataFrame, subset=None) -> pd.DataFrame:
    """
    The function describes amount of duplicated values in each column of dataframe and its percentage
    :param df: dataframe to describe
    :param subset:
    :return: dataframe with two columns that describes number of duplicated rows in each column of dataframe
    """

    if subset:
        num_of_duplicated_rows = df.loc[:, subset].duplicated().sum()
        return pd.DataFrame({"number_of_duplicates": num_of_duplicated_rows,
                             "percentage_of_duplicates": num_of_duplicated_rows / df.size * 100}, index=[str(subset)])

    num_of_duplicated_rows = [df[column].duplicated().sum() for column in df.columns]
    percentage_of_duplicated_rows = [num_rows / df.size * 100 for num_rows in num_of_duplicated_rows]

    return pd.DataFrame(
        {"number_of_duplicates": num_of_duplicated_rows, "percentage_of_duplicates": percentage_of_duplicated_rows},
        index=df.columns)

print('-----Dataframe books_data duplicates before treating duplicates')

print(describe_duplicates(books_data))


print('-----Dataframe books_rating duplicates before treating duplicates')

print(describe_duplicates(books_rating))


# due to number_of_duplicates fir Titles equals to 0, the duplicates
# in dataframe books_data will not be treated

# in dataframe books_rating duplicates treated using the subset "Id", "Title", "User_id"
# trying to avoid users that reviewed the same book with the same Id more than once

books_rating = books_rating.sort_values("review/time")
books_rating = books_rating.drop_duplicates(subset=["Id", "Title", "User_id"], keep="last").reset_index(drop=True)

print('Number of rows in the dataframe books_rating after treating duplicates =', len(books_rating))


print('-----Dataframe books_rating after treating duplicates')
print(describe_duplicates(books_rating))


print(books_rating.groupby(["Id", "Title"])["Id"].count().sort_values(ascending=False).head(10))


# Treating outliers

print('-----Checking outliers')


print('-----Dataframe books_data (ratingsCount)')
print(books_data["ratingsCount"].describe())


fig = px.box(books_data, x="ratingsCount")
fig.update_layout(title_text='ratingsCount')
fig.show()

fig = px.histogram(books_data, x="ratingsCount")
fig.update_layout(title_text='ratingsCount')
fig.show()

print('-----Dataframe books_data (books with ratingsCount > 24)')
print(books_data[books_data["ratingsCount"] > 24]["Title"])


# due to the nature of the outliers in the column is related to famous books, the variable will not be treated

print('-----Dataframe books_rating (Price)')
print(books_rating["Price"].describe())


fig = px.histogram(books_rating, x="Price")
fig.update_traces(marker_color='steelblue', marker_line_color='rgb(80,15,11)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Price')
fig.show()

fig = px.box(books_rating, x="Price")
fig.update_layout(title_text='Price')
fig.show()

print('-----Dataframe books_rating (books with max price)')
print(books_rating.groupby('Title')['Price'].agg('max').sort_values(ascending=False).head(10))


# due to the nature of the outliers in the column is related to expensive books, the variable will not be treated

print('-----Dataframe books_rating (review/score)')
print(books_rating["review/score"].describe())


fig = px.box(books_rating, x="review/score")
fig.update_layout(title_text='review/score')
fig.show()

fig = px.histogram(books_rating, x="review/score")
fig.update_layout(title_text='review/score')
fig.show()

# outliers in the boxplot for the review are not treated due to the needed of created an analysis based on the reviews
# variable review/time is not identified as valuable for the model

# Missing values

print('-----Check NaN and NULL values')

def describe_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    The function describes amount of NA's values in each column of dataframe and its percentage
    :param df: dataframe to describe
    :return: dataframe with two columns that describes number of na rows in each column of dataframe
    """
    df_na = df.isnull().sum()
    df_na_info = pd.concat([df_na, df_na / df.size * 100], axis=1)
    df_na_info.rename(columns={0: "number_of_na", 1: "percentage_of_na"}, inplace=True)
    return df_na_info

print('-----Dataframe books_data')
print(describe_na(books_data))


books_data = books_data.dropna(subset=["Title", "description"])

print('-----Dataframe books_rating')
print(describe_na(books_rating))


books_rating = books_rating.dropna(subset=["Title", "review/summary", "review/text"])

# Most of missing values are in price column, user_id and profileName,
# the last two are not important to achieve the main goals of this project

# check variable price
print('-----Dataframe books_rating (price)')
print(books_rating["Price"].describe())


# replace missing values in variable price with the mean,
# due to this variable could be or not valuable for the prediction models

books_rating["Price"] = books_rating["Price"].fillna(books_rating["Price"].mean())

print('-----Dataframe books_rating after treating missing values')
print(describe_na(books_rating))


data_analysis = DataAnalysis(books_data, books_rating)

#Wordclouds
#cleaning the data

def basic_clean(body):
    """
    Basic function to clean string variables or columns:
        1. Delete duplictaed blank spaces
        2. Everything to lower case
        3. Keep only letters, no numbers or special characters
        4. removes any leading (spaces at the beginning) and trailing (spaces at the end) characters
    """

    body = body.replace(' +', ' ')
    body = body.lower()
    body = body.replace('[^a-z]', "")
    body = body.strip()
    return body

books_rating["review/summary_clean"] = books_rating["review/summary"].apply(lambda x: basic_clean(str(x)))

# Create stopword list
stops = set(STOPWORDS)

# Combine all the reviews into one string
textt = " ".join(x for x in books_rating["review/summary_clean"])

# Genrate the wordcloud

wordcloud = WordCloud(stopwords=stops).generate(textt)
wordcloud.to_file(r"C:\Users\E_jr1\Downloads\wordcloud.png")

# Classify reviews into two 'sentiment' categories called positive and negative

books_rating["sentiment"]=['positive' if x > 3 else 'negative' for x in books_rating["review/score"]]

# categories for the multinomial regression
books_rating["rating_category"]=np.where(books_rating["review/score"]>=4,'Great',np.where(books_rating["review/score"]>=3,'Good',np.where(books_rating["review/score"]>=2,'Fair',np.where(books_rating["review/score"]>=1,'Poor','Bad'))))

print(books_rating.head())

# Check the average values of the score for each sentiment categorie

print(pd.DataFrame(books_rating.groupby(["sentiment"])["review/score"].mean()))

# Generate positive and negative word clouds.

books_rating_p = books_rating[books_rating["sentiment"] == 'positive']
textt_p = " ".join(x for x in books_rating_p["review/summary_clean"])
wordcloud = WordCloud(stopwords=stops).generate(textt_p)
wordcloud.to_file(r"C:\Users\E_jr1\Downloads\wordcloud - positive.png")

books_rating_n = books_rating[books_rating["sentiment"] == 'negative']
textt_n = " ".join(x for x in books_rating_n["review/summary_clean"])
wordcloud = WordCloud(stopwords=stops).generate(textt_n)
wordcloud.to_file(r"C:\Users\E_jr1\Downloads\wordcloud - negative.png")

# sentiment histogram

fig = px.histogram(books_rating, x="sentiment")
fig.update_traces(marker_color="steelblue",marker_line_color='rgb(8,48,96)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Book Sentiment')
fig.show()

# remove stop words
def remove_punc_stopwords(text):
    text = re.sub(f"[{string.punctuation}]"," ",text)
    text_tokens = set(text.split())
    stops = set(stopwords.words('english'))
    text_tokens = text_tokens.difference(stops)
    return " ".join(text_tokens)

books_rating["review/summary_clean"]=books_rating["review/summary_clean"].apply(remove_punc_stopwords)

# split train and test data
books_rating_train = books_rating.sample(frac =.85)
print(len(books_rating_train))
books_rating_test = books_rating.sample(frac =.15)
print(len(books_rating_test))

# count vectorizer due to the lr model does not understand text

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(books_rating_train['review/summary_clean'])
test_matrix = vectorizer.transform(books_rating_test['review/summary_clean'])



# Logistic Regression

lr = LogisticRegression(solver='sag', max_iter=150)

X_train = train_matrix
X_test = test_matrix
y_train = books_rating_train['sentiment']
y_test = books_rating_test['sentiment']

lr.fit(X_train,y_train)

# Generate the predictions for the test dataset
predictions = lr.predict(X_test)
books_rating_test['predictions'] = predictions

#prediction accuracy
books_rating_test['match'] = books_rating_test['sentiment'] == books_rating_test['predictions']
print(sum(books_rating_test['match'])/len(books_rating_test))

# confussion matrix
print(confusion_matrix(predictions,y_test))



# Multinomial Logistic Regression

lr = LogisticRegression(multi_class='multinomial')

X_train = train_matrix
X_test = test_matrix
y_train = books_rating_train['rating_category']
y_test = books_rating_test['rating_category']

lr.fit(X_train,y_train)

# Generate the predictions for the test dataset
predictions = lr.predict(X_test)
books_rating_test['predictions'] = predictions
print(books_rating_test.head(30))

#prediction accuracy
books_rating_test['match'] = books_rating_test['rating_category'] == books_rating_test['predictions']
print(sum(books_rating_test['match'])/len(books_rating_test))

# confussion matrix
print(confusion_matrix(predictions,y_test))