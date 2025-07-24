import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.title("Book Prediction App")
st.info("This app predicts the rating of the inputted book by using the RandomForestRegressor Machine Learning model.")


with st.expander("The Dataset After Initial Cleaning"):
    
    st.write("Dataset")
    
    df = pd.read_csv("https://raw.githubusercontent.com/Kaan-Ince/ML-Book-Rating-Prediction/refs/heads/main/books.csv", sep=",", on_bad_lines="skip", index_col="bookID")
    df.rename({"  num_pages": "num_pages"}, axis=1, inplace=True)

    df["publication_date"] = pd.to_datetime(df["publication_date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna()

    df["language_code"] = df["language_code"].replace(to_replace=['en-US', 'en-GB', 'en-CA', 'enm'], value="eng")

    df = df.drop(df[df["num_pages"] == 0].index)
    df = df.drop(df[df["authors"] == "NOT A BOOK"].index)
    
    df


df["language_code"] = pd.factorize(df["language_code"])[0]
df["publication_date"] = pd.to_datetime(df["publication_date"]).dt.year

df["occurence_count"] = df.groupby("title")["title"].transform("count")

df["occ_weighted"] = df["average_rating"] * df["occurence_count"]
df["rating_count_weighted"] = df["average_rating"] * df["ratings_count"]
df["text_review_count_weighted"] = df["average_rating"] * df["text_reviews_count"]
df["page_weighted"] = df["average_rating"] * df["num_pages"]

df_processed = df.drop(columns=["title", "authors", "isbn", "isbn13", "publisher"])


X = df.drop("average_rating", axis=1)
y = df_processed["average_rating"]

X_processed = df_processed.drop("average_rating", axis=1)

with st.sidebar:
    st.header("Input Book Title")
    title = st.selectbox("Select Book", (df["title"]))
    authors = df["authors"]
    average_rating = df["average_rating"]
    isbn = df["isbn"]
    isbn13 = df["isbn13"]
    language_code = df["language_code"]
    num_pages = df["num_pages"]
    ratings_count = df["ratings_count"]
    text_reviews_count = df["text_reviews_count"]
    publication_date = df["publication_date"]
    publisher = df["publisher"]
    
    input_data = {"title": title, "authors": authors, "average_rating": average_rating,
                  "isbn": isbn, "isbn13": isbn13, "language_code": language_code,
                  "num_pages": num_pages, "ratings_count": ratings_count, "text_reviews_count": text_reviews_count,
                  "publication_date": publication_date, "publisher": publisher}
    input_df = pd.DataFrame(input_data, index=[0])
    input_book = pd.concat([input_df, X], axis=0)

input_row = input_book[:1]
input_rating = input_row.average_rating

random_forest_reg = RandomForestRegressor(random_state=42)
random_forest_reg.fit(X_processed, y)
y_predicted = np.round(random_forest_reg.predict(input_rating), 2)


st.subheader("Predicted Average Rating")
y_predicted
