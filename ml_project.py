import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv("C:/Users/Kaan/Documents/Masters/(Week 4) Python Machine Learning Labs/Project 1/books.csv", sep=",", on_bad_lines="skip", index_col="bookID")


df.rename({"  num_pages": "num_pages"}, axis=1, inplace=True)

df["publication_date"] = pd.to_datetime(df["publication_date"], format="%m/%d/%Y", errors="coerce")
df = df.dropna()

df["language_code"] = df["language_code"].replace(to_replace=['en-US', 'en-GB', 'en-CA', 'enm'], value="eng")

df = df.drop(df[df["num_pages"] == 0].index)
df = df.drop(df[df["authors"] == "NOT A BOOK"].index)


df["language_code"] = pd.factorize(df["language_code"])[0]
df["publication_date"] = pd.to_datetime(df["publication_date"]).dt.year


df["occurence_count"] = df.groupby("title")["title"].transform("count")

df["occ_weighted"] = df["average_rating"] * df["occurence_count"]
df["rating_count_weighted"] = df["average_rating"] * df["ratings_count"]
df["text_review_count_weighted"] = df["average_rating"] * df["text_reviews_count"]
df["page_weighted"] = df["average_rating"] * df["num_pages"]

df_processed = df.drop(columns=["title", "authors", "isbn", "isbn13", "publisher"])

df_train, df_test = train_test_split(df_processed, test_size=0.20, random_state=42)

X_train = df_train.drop("average_rating", axis=1)
y_train = df_train["average_rating"]

X_test = df_test.drop("average_rating", axis=1)
y_test = df_test["average_rating"]


random_forest_reg = RandomForestRegressor(random_state=42)
random_forest_reg.fit(X_train, y_train)
y_predicted = np.round(random_forest_reg.predict(X_test), 2)


accuracy = (y_predicted == y_test).sum()/len(y_test)
score = np.round(r2_score(y_test, y_predicted) * 100, 2)
error = np.sqrt(mean_squared_error(y_test, y_predicted))

print("Accuracy calculated by hand:", np.round((accuracy * 100), 2), "%")
print("R2 Score:", score)
print("Mean Squared Error:", np.round(error, 3))
