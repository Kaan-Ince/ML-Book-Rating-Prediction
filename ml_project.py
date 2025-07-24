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


df2 = df

df2["language_code"] = pd.factorize(df2["language_code"])[0]
df2["publication_date"] = pd.to_datetime(df2["publication_date"]).dt.year

df2["occurence_count"] = df2.groupby("title")["title"].transform("count")

df2["occ_weighted"] = df2["average_rating"] * df2["occurence_count"]
df2["rating_count_weighted"] = df2["average_rating"] * df2["ratings_count"]
df2["text_review_count_weighted"] = df2["average_rating"] * df2["text_reviews_count"]
df2["page_weighted"] = df2["average_rating"] * df2["num_pages"]

df_processed = df2.drop(columns=["authors", "isbn", "isbn13", "publisher"])


X = df.drop("average_rating", axis=1)
y = df_processed["average_rating"]
X_processed = df_processed.drop("average_rating", axis=1)

with st.sidebar:
    st.header("Input Book Title")
    title = st.selectbox("Select Book", (df["title"]))
    input_data = df.loc[df["title"] == title]
    input_processed = df_processed.loc[df_processed["title"] == title]

input_row = input_data[:1]

with st.expander("Input Row"):
    input_row

input_prediction = df_processed.drop(columns=["average_rating", "title"])

X_final = df_processed.drop("title", axis=1)

random_forest_reg = RandomForestRegressor(random_state=42)
random_forest_reg.fit(X_final, y)
y_predicted = np.round(random_forest_reg.predict(input_prediction), 2)

with st.expander("Prediction"):
    st.subheader("Predicted Average Rating")
    y_predicted
