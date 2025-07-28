# ML-Book-Rating-Prediction

Machine Learning project that predicts the average rating of a book in a 0-5 star range. Dataset is obtained from Goodreads.

Below link is the app built with Streamlit that showcases the prediction of the model with user input that can be entered in the sidebar.<br />

[Book Rating Prediction App](https://ml-book-rating-prediction-axpmpwvjb4nkfnqy4baljd.streamlit.app/)





## 1. Data Preparation Stage




```python
df.nunique()
```




    title                 10348
    authors                6639
    average_rating          209
    isbn                  11123
    isbn13                11123
    language_code            27
    num_pages               997
    ratings_count          5294
    text_reviews_count     1822
    publication_date       3679
    publisher              2290
    dtype: int64



#### !!! Above result gives us 10348 unique book titles, while the dataset includes 11123 books in total !!!
This is addressed and taken into account during the Fine-Tuning stage.





```python
df["publication_date"] = pd.to_datetime(df["publication_date"], format="%m/%d/%Y", errors="coerce")
```


```python
df.isna().sum()
```




    title                 0
    authors               0
    average_rating        0
    isbn                  0
    isbn13                0
    language_code         0
    num_pages             0
    ratings_count         0
    text_reviews_count    0
    publication_date      2
    publisher             0
    dtype: int64



We immediately remove the new null values caused by the datetime transformation.


```python
df = df.dropna()
```


```python
df.isna().sum()
```




    title                 0
    authors               0
    average_rating        0
    isbn                  0
    isbn13                0
    language_code         0
    num_pages             0
    ratings_count         0
    text_reviews_count    0
    publication_date      0
    publisher             0
    dtype: int64



We can group the regional (and "enm" as Middle English) English language codes as a whole under "eng".


```python
df.language_code.unique()
```




    array(['eng', 'en-US', 'fre', 'spa', 'en-GB', 'mul', 'grc', 'enm',
           'en-CA', 'ger', 'jpn', 'ara', 'nl', 'zho', 'lat', 'por', 'srp',
           'ita', 'rus', 'msa', 'glg', 'wel', 'swe', 'nor', 'tur', 'gla',
           'ale'], dtype=object)




```python
df["language_code"] = df["language_code"].replace(['en-US', 'en-GB', 'en-CA', 'enm'], "eng")
```

    


```python
df.language_code.unique()
```




    array(['eng', 'fre', 'spa', 'mul', 'grc', 'ger', 'jpn', 'ara', 'nl',
           'zho', 'lat', 'por', 'srp', 'ita', 'rus', 'msa', 'glg', 'wel',
           'swe', 'nor', 'tur', 'gla', 'ale'], dtype=object)






There are books with zero number of pages. Below, we replace these zero values with the mean page numbers with respect to the language that the book is in.





There 76 such books. Considering the size of our dataset, it should be safe to drop these rows without significantly affecting model performance.


```python
df = df.drop(df[df["num_pages"] == 0].index)
```

We now drop the rows with the entry "NOT A BOOK" in the Authors column.







```python
df = df.drop(df[df["authors"] == "NOT A BOOK"].index)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 11042 entries, 1 to 45641
    Data columns (total 11 columns):
     #   Column              Non-Null Count  Dtype         
    ---  ------              --------------  -----         
     0   title               11042 non-null  object        
     1   authors             11042 non-null  object        
     2   average_rating      11042 non-null  float64       
     3   isbn                11042 non-null  object        
     4   isbn13              11042 non-null  int64         
     5   language_code       11042 non-null  object        
     6   num_pages           11042 non-null  int64         
     7   ratings_count       11042 non-null  int64         
     8   text_reviews_count  11042 non-null  int64         
     9   publication_date    11042 non-null  datetime64[ns]
     10  publisher           11042 non-null  object        
    dtypes: datetime64[ns](1), float64(1), int64(4), object(5)
    memory usage: 1.0+ MB
    

## 2. EDA Stage


### Average Rating v Rating Count Scatter Plot




    




    
![alt text](https://github.com/Kaan-Ince/ML-Book-Rating-Prediction/blob/main/plots/output_30_1.png "Plot 2.1")
    





### Average Rating v Text Review Count Scatter Plot

    




    
![alt text](https://github.com/Kaan-Ince/ML-Book-Rating-Prediction/blob/main/plots/output_31_1.png "Plot 2.2")
    



### Top 5 Languages and their percantages in the dataset


    
![alt text](https://github.com/Kaan-Ince/ML-Book-Rating-Prediction/blob/main/plots/output_32_0.png "Plot 2.3")
    



### Average Rating Distribution




    




    
![alt text](https://github.com/Kaan-Ince/ML-Book-Rating-Prediction/blob/main/plots/output_33_1.png "Plot 2.4")
    



### Ratings Count Distribution




    




    
![alt text](https://github.com/Kaan-Ince/ML-Book-Rating-Prediction/blob/main/plots/output_34_1.png "Plot 2.5")
    



### Text Review Count Distribution




    




    
![alt text](https://github.com/Kaan-Ince/ML-Book-Rating-Prediction/blob/main/plots/output_35_1.png "Plot 2.6")
    


Below are some visualizations of top 10 books with respect to different criteria.


### Top 10 Books With Highest Average Rating

    
    


    
![alt text](https://github.com/Kaan-Ince/ML-Book-Rating-Prediction/blob/main/plots/output_38_1.png "Plot 2.6")
    



### Top 10 Books with Highest Rating Count

    
    


    
![alt text](https://github.com/Kaan-Ince/ML-Book-Rating-Prediction/blob/main/plots/output_40_1.png "Plot 2.7")
    



### Top 10 Books with Highest Text Review Count


    
    


    
![alt text](https://github.com/Kaan-Ince/ML-Book-Rating-Prediction/blob/main/plots/output_42_1.png "Plot 2.8")
    



### Top 10 Longest Books

    
    


    
![alt text](https://github.com/Kaan-Ince/ML-Book-Rating-Prediction/blob/main/plots/output_44_1.png "Plot 2.6")
    


## 3. Feature Engineering Stage

We first need to transform the language column data to numeric.


```python
df["language_code"] = pd.factorize(df["language_code"])[0]
```

In order to transform the publication date column to numeric values, we can extract the year of a book's publication (also since it is the most relevant to us).


```python
df["publication_date"] = pd.to_datetime(df["publication_date"]).dt.year
```


```python
df_processed = df.drop(columns=["title", "authors", "isbn", "isbn13", "publisher"])
```







### Correlation Heatmap


    
![alt text](https://github.com/Kaan-Ince/ML-Book-Rating-Prediction/blob/main/plots/output_52_0.png "Plot 3.1")
    


## 4. Modeling Stage



```python
df_train, df_test = train_test_split(df_processed, test_size=0.20, random_state=42)
```



```python
X_train = df_train.drop("average_rating", axis=1)
y_train = df_train["average_rating"]
```


```python
X_test = df_test.drop("average_rating", axis=1)
y_test = df_test["average_rating"]
```

Since our aim is to predict a float value between 0 and 5, Regressor models are our best choices.





```python
models = {"LinearRegression": LinearRegression(),
             "DecisionTree": DecisionTreeRegressor(random_state=42), "RandomForest": RandomForestRegressor(random_state=42),
             "GradientBoosting": GradientBoostingRegressor(random_state=42),
             "KNeighbors": KNeighborsRegressor()}
```


```python
results = {}
mean_sq_error = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_predicted = np.round(model.predict(X_test), 2)
    score = r2_score(y_test, y_predicted)
    error = np.sqrt(mean_squared_error(y_test, y_predicted))
    results[name] = score
    mean_sq_error[name] = error
```





### Regressor Model Accuracy (R2 Score)

    
    


    
![alt text](https://github.com/Kaan-Ince/ML-Book-Rating-Prediction/blob/main/plots/output_64_1.png "Plot 4.1")
    


As can be seen from above plot, the models performed poorly. We can go back to the Feature Engineering stage and add new feature to improve performance.

## 5. Fine-Tuning Stage

Let's "copy" our dataframe with a different name.


```python
df2 = df
```

Recall that in the data preparation stage, we noticed that the number of UNIQUE book titles is lower than total number of books in the dataset. So, we can see the number of times a book is entered in the dataset. These different occurences are most likely caused by:
1. Different language of the book
2. Different publisher
3. Different edition
4. Multiple Authors (editors, translators)

Hence, we add a new column to our dataframe that stores the number of occurences each unique book has in total with the below code.


```python
df2["occurence_count"] = df2.groupby("title")["title"].transform("count")
```


```python
df2.occurence_count.value_counts(ascending=True)
```




    occurence_count
    8      16
    9      18
    7      28
    6      66
    5      95
    4     120
    3     225
    2     670
    1    9804
    Name: count, dtype: int64



By above, we can see that, for example, 18 books can be seen in 9 different rows.

Next we add new features. Namely, we do four multiplications by the average rating of each book with their occurence count, rating and text review count, and page number respectively to get four total new columns in our dataframe to feed into our models.


```python
df2["occ_weighted"] = df2["average_rating"] * df2["occurence_count"]
```


```python
df2["rating_count_weighted"] = df2["average_rating"] * df2["ratings_count"]
df2["text_review_count_weighted"] = df2["average_rating"] * df2["text_reviews_count"]
```


```python
df2["page_weighted"] = df2["average_rating"] * df2["num_pages"]
```




After dropping non-numeric columns, the dataframe is ready to be fitted into the models again.


```python
df2_processed = df2.drop(columns=["title", "authors", "isbn", "isbn13", "publisher"])
```




### Correlation Heatmap of the revised dataframe




    
![alt text](https://github.com/Kaan-Ince/ML-Book-Rating-Prediction/blob/main/plots/output_82_0.png "Plot 5.1")
    



```python
df2_train, df2_test = train_test_split(df2_processed, test_size=0.20, random_state=42)

X_train2 = df2_train.drop("average_rating", axis=1)
y_train2 = df2_train["average_rating"]

X_test2 = df2_test.drop("average_rating", axis=1)
y_test2 = df2_test["average_rating"]
```

    


```python
results2 = {}
mean_sq_error2 = {}

for name, model in models.items():
    model.fit(X_train2, y_train2)
    y_predicted = np.round(model.predict(X_test2),2)
    score = r2_score(y_test2, y_predicted)
    error = np.sqrt(mean_squared_error(y_test2, y_predicted))
    results2[name] = score
    mean_sq_error2[name] = error
```




### Regressor Model Accuracy (R2 Score) - Revised

    


    
![alt text](https://github.com/Kaan-Ince/ML-Book-Rating-Prediction/blob/main/plots/output_87_1.png "Plot 5.2")
    


## 6. Conclusion

As can be seen from the above plot, models performed significantly better after the addition of the new features, with three of our five models performing in the 99.33 - 99.45 range. Furthermore, the RandomForestRegressor model performed the best in this iteration with an R2 score of 99.44% (when rounded to two decimals). Therefore, we select it for our main model to be used in the final application.



#### Final Model Evaluation for RandomForestRegressor


```python
accuracy = (y_predicted == y_test2).sum()/len(y_test2)
```

Accuracy of RandomForestRegressor calculated by hand: 93.39 %
    


```python
score = r2_score(y_test2, y_predicted)
```

R2 Score of RandomForestRegressor: 99.44
    


```python
error = np.sqrt(mean_squared_error(y_test2, y_predicted))
```

MSE for RandomForestRegressor: 0.027
    
