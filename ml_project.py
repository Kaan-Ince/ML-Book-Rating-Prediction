import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

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

df_processed = df.drop(columns=["authors", "isbn", "isbn13", "publisher"])


X = df_processed.drop(columns=["title", "average_rating"], axis=1)
y = df_processed["average_rating"]


with st.sidebar:
    st.header("Input Book Title")
    title = st.selectbox("Select Book", (df["title"]))
    input_data = df.loc[df["title"] == title]
    input_processed = df_processed.loc[df_processed["title"] == title]


input_row = input_data[:1]
input_row_processed = input_processed[:1]

with st.expander("Input Row"):
    input_row

input_prediction = input_row_processed.drop(columns=["title", "average_rating"], axis=1)


random_forest_reg = RandomForestRegressor(random_state=42)
random_forest_reg.fit(X, y)
y_predicted = np.round(random_forest_reg.predict(input_prediction), 2)

with st.expander("Prediction"):
    st.subheader("Predicted Average Rating")
    y_predicted
