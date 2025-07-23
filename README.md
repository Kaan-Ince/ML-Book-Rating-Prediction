# ML-Book-Rating-Prediction

Machine Learning project that predicts the average rating of a book in a 0-5 star range. Dataset is obtained from Goodreads.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
df = pd.read_csv("./books.csv", sep=",", on_bad_lines="skip", index_col="bookID")
```


```python
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>authors</th>
      <th>average_rating</th>
      <th>isbn</th>
      <th>isbn13</th>
      <th>language_code</th>
      <th>num_pages</th>
      <th>ratings_count</th>
      <th>text_reviews_count</th>
      <th>publication_date</th>
      <th>publisher</th>
    </tr>
    <tr>
      <th>bookID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Harry Potter and the Half-Blood Prince (Harry ...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.57</td>
      <td>0439785960</td>
      <td>9780439785969</td>
      <td>eng</td>
      <td>652</td>
      <td>2095690</td>
      <td>27591</td>
      <td>9/16/2006</td>
      <td>Scholastic Inc.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Harry Potter and the Order of the Phoenix (Har...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.49</td>
      <td>0439358078</td>
      <td>9780439358071</td>
      <td>eng</td>
      <td>870</td>
      <td>2153167</td>
      <td>29221</td>
      <td>9/1/2004</td>
      <td>Scholastic Inc.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Harry Potter and the Chamber of Secrets (Harry...</td>
      <td>J.K. Rowling</td>
      <td>4.42</td>
      <td>0439554896</td>
      <td>9780439554893</td>
      <td>eng</td>
      <td>352</td>
      <td>6333</td>
      <td>244</td>
      <td>11/1/2003</td>
      <td>Scholastic</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Harry Potter and the Prisoner of Azkaban (Harr...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.56</td>
      <td>043965548X</td>
      <td>9780439655484</td>
      <td>eng</td>
      <td>435</td>
      <td>2339585</td>
      <td>36325</td>
      <td>5/1/2004</td>
      <td>Scholastic Inc.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Harry Potter Boxed Set  Books 1-5 (Harry Potte...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.78</td>
      <td>0439682584</td>
      <td>9780439682589</td>
      <td>eng</td>
      <td>2690</td>
      <td>41428</td>
      <td>164</td>
      <td>9/13/2004</td>
      <td>Scholastic</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>45631</th>
      <td>Expelled from Eden: A William T. Vollmann Reader</td>
      <td>William T. Vollmann/Larry McCaffery/Michael He...</td>
      <td>4.06</td>
      <td>1560254416</td>
      <td>9781560254416</td>
      <td>eng</td>
      <td>512</td>
      <td>156</td>
      <td>20</td>
      <td>12/21/2004</td>
      <td>Da Capo Press</td>
    </tr>
    <tr>
      <th>45633</th>
      <td>You Bright and Risen Angels</td>
      <td>William T. Vollmann</td>
      <td>4.08</td>
      <td>0140110879</td>
      <td>9780140110876</td>
      <td>eng</td>
      <td>635</td>
      <td>783</td>
      <td>56</td>
      <td>12/1/1988</td>
      <td>Penguin Books</td>
    </tr>
    <tr>
      <th>45634</th>
      <td>The Ice-Shirt (Seven Dreams #1)</td>
      <td>William T. Vollmann</td>
      <td>3.96</td>
      <td>0140131965</td>
      <td>9780140131963</td>
      <td>eng</td>
      <td>415</td>
      <td>820</td>
      <td>95</td>
      <td>8/1/1993</td>
      <td>Penguin Books</td>
    </tr>
    <tr>
      <th>45639</th>
      <td>Poor People</td>
      <td>William T. Vollmann</td>
      <td>3.72</td>
      <td>0060878827</td>
      <td>9780060878825</td>
      <td>eng</td>
      <td>434</td>
      <td>769</td>
      <td>139</td>
      <td>2/27/2007</td>
      <td>Ecco</td>
    </tr>
    <tr>
      <th>45641</th>
      <td>Las aventuras de Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>3.91</td>
      <td>8497646983</td>
      <td>9788497646987</td>
      <td>spa</td>
      <td>272</td>
      <td>113</td>
      <td>12</td>
      <td>5/28/2006</td>
      <td>Edimat Libros</td>
    </tr>
  </tbody>
</table>
<p>11123 rows × 11 columns</p>
</div>



## 1. Data Preparation Stage


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 11123 entries, 1 to 45641
    Data columns (total 11 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   title               11123 non-null  object 
     1   authors             11123 non-null  object 
     2   average_rating      11123 non-null  float64
     3   isbn                11123 non-null  object 
     4   isbn13              11123 non-null  int64  
     5   language_code       11123 non-null  object 
     6     num_pages         11123 non-null  int64  
     7   ratings_count       11123 non-null  int64  
     8   text_reviews_count  11123 non-null  int64  
     9   publication_date    11123 non-null  object 
     10  publisher           11123 non-null  object 
    dtypes: float64(1), int64(4), object(6)
    memory usage: 1.0+ MB
    


```python
df.rename({"  num_pages": "num_pages"}, axis=1, inplace=True)
```


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
df.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>average_rating</th>
      <th>isbn13</th>
      <th>num_pages</th>
      <th>ratings_count</th>
      <th>text_reviews_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11123.000000</td>
      <td>1.112300e+04</td>
      <td>11123.000000</td>
      <td>1.112300e+04</td>
      <td>11123.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.934075</td>
      <td>9.759880e+12</td>
      <td>336.405556</td>
      <td>1.794285e+04</td>
      <td>542.048099</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.350485</td>
      <td>4.429758e+11</td>
      <td>241.152626</td>
      <td>1.124992e+05</td>
      <td>2576.619589</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>8.987060e+09</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.770000</td>
      <td>9.780345e+12</td>
      <td>192.000000</td>
      <td>1.040000e+02</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.960000</td>
      <td>9.780582e+12</td>
      <td>299.000000</td>
      <td>7.450000e+02</td>
      <td>47.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.140000</td>
      <td>9.780872e+12</td>
      <td>416.000000</td>
      <td>5.000500e+03</td>
      <td>238.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>9.790008e+12</td>
      <td>6576.000000</td>
      <td>4.597666e+06</td>
      <td>94265.000000</td>
    </tr>
  </tbody>
</table>
</div>




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




```python
df.language_code.value_counts()
```




    language_code
    eng    10539
    spa      218
    fre      143
    ger       99
    jpn       46
    mul       19
    zho       14
    grc       11
    por       10
    ita        5
    lat        3
    rus        2
    swe        2
    ara        1
    nl         1
    srp        1
    msa        1
    glg        1
    wel        1
    nor        1
    tur        1
    gla        1
    ale        1
    Name: count, dtype: int64



There are books with zero number of pages. Below, we replace these zero values with the mean page numbers with respect to the language that the book is in.


```python
df.loc[df["num_pages"] == 0]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>authors</th>
      <th>average_rating</th>
      <th>isbn</th>
      <th>isbn13</th>
      <th>language_code</th>
      <th>num_pages</th>
      <th>ratings_count</th>
      <th>text_reviews_count</th>
      <th>publication_date</th>
      <th>publisher</th>
    </tr>
    <tr>
      <th>bookID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>955</th>
      <td>The 5 Love Languages / The 5 Love Languages Jo...</td>
      <td>Gary Chapman</td>
      <td>4.70</td>
      <td>0802415318</td>
      <td>9780802415318</td>
      <td>eng</td>
      <td>0</td>
      <td>22</td>
      <td>4</td>
      <td>2005-01-01</td>
      <td>Moody Publishers</td>
    </tr>
    <tr>
      <th>2835</th>
      <td>The Tragedy of Pudd'nhead Wilson</td>
      <td>Mark Twain/Michael Prichard</td>
      <td>3.79</td>
      <td>140015068X</td>
      <td>9781400150687</td>
      <td>eng</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2003-01-01</td>
      <td>Tantor Media</td>
    </tr>
    <tr>
      <th>3593</th>
      <td>Murder by Moonlight &amp; Other Mysteries (New Adv...</td>
      <td>NOT A BOOK</td>
      <td>4.00</td>
      <td>0743564677</td>
      <td>9780743564670</td>
      <td>eng</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
      <td>2006-10-03</td>
      <td>Simon  Schuster Audio</td>
    </tr>
    <tr>
      <th>3599</th>
      <td>The Unfortunate Tobacconist &amp; Other Mysteries ...</td>
      <td>NOT A BOOK</td>
      <td>3.50</td>
      <td>074353395X</td>
      <td>9780743533959</td>
      <td>eng</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>2003-10-01</td>
      <td>Simon &amp; Schuster Audio</td>
    </tr>
    <tr>
      <th>4249</th>
      <td>The Da Vinci Code (Robert Langdon  #2)</td>
      <td>Dan Brown/Paul Michael</td>
      <td>3.84</td>
      <td>0739339788</td>
      <td>9780739339787</td>
      <td>eng</td>
      <td>0</td>
      <td>91</td>
      <td>16</td>
      <td>2006-03-28</td>
      <td>Random House Audio</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>40378</th>
      <td>The Chessmen of Mars (Barsoom #5)</td>
      <td>Edgar Rice Burroughs/John Bolen</td>
      <td>3.83</td>
      <td>1400130212</td>
      <td>9781400130214</td>
      <td>eng</td>
      <td>0</td>
      <td>5147</td>
      <td>157</td>
      <td>2005-01-01</td>
      <td>Tantor Media</td>
    </tr>
    <tr>
      <th>41273</th>
      <td>Fine Lines (One-Eyed Mack  #6)</td>
      <td>Jim Lehrer</td>
      <td>3.23</td>
      <td>0517164353</td>
      <td>9780517164358</td>
      <td>eng</td>
      <td>0</td>
      <td>17</td>
      <td>4</td>
      <td>1995-11-19</td>
      <td>Random House Value Publishing</td>
    </tr>
    <tr>
      <th>43343</th>
      <td>Stowaway and Milk Run: Two Unabridged Stories ...</td>
      <td>Mary Higgins Clark/Jan Maxwell</td>
      <td>3.49</td>
      <td>0671046241</td>
      <td>9780671046248</td>
      <td>eng</td>
      <td>0</td>
      <td>64</td>
      <td>2</td>
      <td>1999-12-01</td>
      <td>Simon &amp; Schuster Audio</td>
    </tr>
    <tr>
      <th>44748</th>
      <td>The Mask of the Enchantress</td>
      <td>Victoria Holt</td>
      <td>3.85</td>
      <td>0449210847</td>
      <td>9780449210840</td>
      <td>eng</td>
      <td>0</td>
      <td>21</td>
      <td>1</td>
      <td>1981-10-12</td>
      <td>Ivy Books</td>
    </tr>
    <tr>
      <th>45472</th>
      <td>Treasury of American Tall Tales: Volume 1: Dav...</td>
      <td>David Bromberg/Jay Ungar/Molly Mason/Garrison ...</td>
      <td>3.86</td>
      <td>0739336509</td>
      <td>9780739336502</td>
      <td>eng</td>
      <td>0</td>
      <td>36</td>
      <td>9</td>
      <td>2006-08-22</td>
      <td>Listening Library (Audio)</td>
    </tr>
  </tbody>
</table>
<p>76 rows × 11 columns</p>
</div>



There 76 such books. Considering the size of our dataset, it should be safe to drop these rows without significantly affecting model performance.


```python
df = df.drop(df[df["num_pages"] == 0].index)
```

We now drop the rows with the entry "NOT A BOOK" in the Authors column.


```python
df.loc[df["authors"] == "NOT A BOOK"]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>authors</th>
      <th>average_rating</th>
      <th>isbn</th>
      <th>isbn13</th>
      <th>language_code</th>
      <th>num_pages</th>
      <th>ratings_count</th>
      <th>text_reviews_count</th>
      <th>publication_date</th>
      <th>publisher</th>
    </tr>
    <tr>
      <th>bookID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19786</th>
      <td>The Goon Show  Volume 4: My Knees Have Fallen ...</td>
      <td>NOT A BOOK</td>
      <td>5.00</td>
      <td>0563388692</td>
      <td>9780563388692</td>
      <td>eng</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>1996-04-01</td>
      <td>BBC Physical Audio</td>
    </tr>
    <tr>
      <th>19787</th>
      <td>The Goon Show: Moriarty Where Are You?</td>
      <td>NOT A BOOK</td>
      <td>4.43</td>
      <td>0563388544</td>
      <td>9780563388548</td>
      <td>eng</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2005-03-30</td>
      <td>BBC Physical Audio</td>
    </tr>
    <tr>
      <th>19788</th>
      <td>The Goon Show  Volume 11: He's Fallen in the W...</td>
      <td>NOT A BOOK</td>
      <td>5.00</td>
      <td>0563388323</td>
      <td>9780563388326</td>
      <td>eng</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1995-10-02</td>
      <td>BBC Physical Audio</td>
    </tr>
  </tbody>
</table>
</div>




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


```python
plt.figure(figsize=(10,5))
plt.scatter(df.average_rating, df.ratings_count)
plt.title("Rating vs Rating Count")
plt.xlabel("Rating")
plt.ylabel("Rating Count")
plt.show
```




    




    
![png](output_30_1.png)
    



```python
plt.figure(figsize=(10,5))
plt.scatter(df.average_rating, df.text_reviews_count)
plt.title("Rating vs Text Review Count")
plt.xlabel("Rating")
plt.ylabel("Text Review Count")
plt.show
```




    




    
![png](output_31_1.png)
    



```python
plt.pie(df.language_code.value_counts().nlargest(5),labels=df.language_code.value_counts().nlargest(5).index,autopct='%1.2f%%')
plt.show()
```


    
![png](output_32_0.png)
    



```python
sns.histplot(df.average_rating, kde=True)
```




    




    
![png](output_33_1.png)
    



```python
sns.histplot(df.ratings_count, kde=True)
```




    




    
![png](output_34_1.png)
    



```python
sns.histplot(df.text_reviews_count, kde=True)
```




    




    
![png](output_35_1.png)
    


Below are some visualizations of top 10 books with respect to different criteria.


```python
top10_books = df.sort_values(by="average_rating", ascending=False).head(10)
```


```python
plt.figure(figsize=(12, 6))
sns.barplot(x="average_rating", y="title", data=top10_books, palette="mako")
plt.xlabel("Average Rating")
plt.ylabel("Book Title")
plt.title("Top 10 Books With Highest Average Rating")
plt.xlim(0, 5)
plt.show()
```

    
    


    
![png](output_38_1.png)
    



```python
top10_ratingcount = df.sort_values(by="ratings_count", ascending=False).head(10)
```


```python
plt.figure(figsize=(12, 6))
sns.barplot(x="ratings_count", y="title", data=top10_ratingcount, palette="crest")
plt.xlabel("Rating Count")
plt.ylabel("Book Title")
plt.title("Top 10 Books with Highest Rating Count")
plt.show()
```

    
    


    
![png](output_40_1.png)
    



```python
top10_textreviewcount = df.sort_values(by="text_reviews_count", ascending=False).head(10)
```


```python
plt.figure(figsize=(12, 6))
sns.barplot(x="text_reviews_count", y="title", data=top10_textreviewcount, palette="viridis")
plt.xlabel("Text Review Count")
plt.ylabel("Book Title")
plt.title("Top 10 Books with Highest Text Review Count")
plt.show()
```

    
    


    
![png](output_42_1.png)
    



```python
top10_longest = df.sort_values(by="num_pages", ascending=False).head(10)
```


```python
plt.figure(figsize=(12, 6))
sns.barplot(x="num_pages", y="title", data=top10_longest, palette="flare")
plt.xlabel("Page Number")
plt.ylabel("Book Title")
plt.title("Top 10 Longest Books")
plt.show()
```

    
    


    
![png](output_44_1.png)
    


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


```python
df_processed
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>average_rating</th>
      <th>language_code</th>
      <th>num_pages</th>
      <th>ratings_count</th>
      <th>text_reviews_count</th>
      <th>publication_date</th>
    </tr>
    <tr>
      <th>bookID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4.57</td>
      <td>0</td>
      <td>652</td>
      <td>2095690</td>
      <td>27591</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.49</td>
      <td>0</td>
      <td>870</td>
      <td>2153167</td>
      <td>29221</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.42</td>
      <td>0</td>
      <td>352</td>
      <td>6333</td>
      <td>244</td>
      <td>2003</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.56</td>
      <td>0</td>
      <td>435</td>
      <td>2339585</td>
      <td>36325</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4.78</td>
      <td>0</td>
      <td>2690</td>
      <td>41428</td>
      <td>164</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>45631</th>
      <td>4.06</td>
      <td>0</td>
      <td>512</td>
      <td>156</td>
      <td>20</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>45633</th>
      <td>4.08</td>
      <td>0</td>
      <td>635</td>
      <td>783</td>
      <td>56</td>
      <td>1988</td>
    </tr>
    <tr>
      <th>45634</th>
      <td>3.96</td>
      <td>0</td>
      <td>415</td>
      <td>820</td>
      <td>95</td>
      <td>1993</td>
    </tr>
    <tr>
      <th>45639</th>
      <td>3.72</td>
      <td>0</td>
      <td>434</td>
      <td>769</td>
      <td>139</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>45641</th>
      <td>3.91</td>
      <td>2</td>
      <td>272</td>
      <td>113</td>
      <td>12</td>
      <td>2006</td>
    </tr>
  </tbody>
</table>
<p>11042 rows × 6 columns</p>
</div>




```python
plt.figure(figsize=(10,8))
sns.heatmap(df_processed.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.show()
```


    
![png](output_52_0.png)
    


## 4. Modeling Stage


```python
from sklearn.model_selection import train_test_split
```


```python
df_train, df_test = train_test_split(df_processed, test_size=0.20, random_state=42)
```


```python
print(len(df_processed))
print(len(df_train))
print(len(df_test))
```

    11042
    8833
    2209
    


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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
```


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


```python
print(results.values())
print(mean_sq_error.values())
```

    dict_values([0.02797054261915699, -0.5642721312335328, 0.10214538317383848, 0.18262676787549703, -0.09455041215495963])
    dict_values([np.float64(0.3505802601213898), np.float64(0.44473765289258727), np.float64(0.3369385912968644), np.float64(0.32148295472163135), np.float64(0.3720194361078598)])
    


```python
plt.figure(figsize=(10, 6))
sns.barplot(x=list(results.values()), y=list(results.keys()), palette='viridis')
plt.xlabel('R2 Score')
plt.title('Regressor Model Accuracy (R2 Score)')
plt.xlim(0, 1)
plt.show()
```

    
    


    
![png](output_64_1.png)
    


As can be seen from above plot, the models performed poorly. We can go back to the Feature Engineering stage and add new feature to improve performance.

### 5. Fine-Tuning Stage

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


```python
df2
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>authors</th>
      <th>average_rating</th>
      <th>isbn</th>
      <th>isbn13</th>
      <th>language_code</th>
      <th>num_pages</th>
      <th>ratings_count</th>
      <th>text_reviews_count</th>
      <th>publication_date</th>
      <th>publisher</th>
      <th>occurence_count</th>
      <th>occ_weighted</th>
      <th>rating_count_weighted</th>
      <th>text_review_count_weighted</th>
      <th>page_weighted</th>
    </tr>
    <tr>
      <th>bookID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Harry Potter and the Half-Blood Prince (Harry ...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.57</td>
      <td>0439785960</td>
      <td>9780439785969</td>
      <td>0</td>
      <td>652</td>
      <td>2095690</td>
      <td>27591</td>
      <td>2006</td>
      <td>Scholastic Inc.</td>
      <td>2</td>
      <td>9.14</td>
      <td>9577303.30</td>
      <td>126090.87</td>
      <td>2979.64</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Harry Potter and the Order of the Phoenix (Har...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.49</td>
      <td>0439358078</td>
      <td>9780439358071</td>
      <td>0</td>
      <td>870</td>
      <td>2153167</td>
      <td>29221</td>
      <td>2004</td>
      <td>Scholastic Inc.</td>
      <td>1</td>
      <td>4.49</td>
      <td>9667719.83</td>
      <td>131202.29</td>
      <td>3906.30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Harry Potter and the Chamber of Secrets (Harry...</td>
      <td>J.K. Rowling</td>
      <td>4.42</td>
      <td>0439554896</td>
      <td>9780439554893</td>
      <td>0</td>
      <td>352</td>
      <td>6333</td>
      <td>244</td>
      <td>2003</td>
      <td>Scholastic</td>
      <td>2</td>
      <td>8.84</td>
      <td>27991.86</td>
      <td>1078.48</td>
      <td>1555.84</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Harry Potter and the Prisoner of Azkaban (Harr...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.56</td>
      <td>043965548X</td>
      <td>9780439655484</td>
      <td>0</td>
      <td>435</td>
      <td>2339585</td>
      <td>36325</td>
      <td>2004</td>
      <td>Scholastic Inc.</td>
      <td>2</td>
      <td>9.12</td>
      <td>10668507.60</td>
      <td>165642.00</td>
      <td>1983.60</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Harry Potter Boxed Set  Books 1-5 (Harry Potte...</td>
      <td>J.K. Rowling/Mary GrandPré</td>
      <td>4.78</td>
      <td>0439682584</td>
      <td>9780439682589</td>
      <td>0</td>
      <td>2690</td>
      <td>41428</td>
      <td>164</td>
      <td>2004</td>
      <td>Scholastic</td>
      <td>1</td>
      <td>4.78</td>
      <td>198025.84</td>
      <td>783.92</td>
      <td>12858.20</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>45631</th>
      <td>Expelled from Eden: A William T. Vollmann Reader</td>
      <td>William T. Vollmann/Larry McCaffery/Michael He...</td>
      <td>4.06</td>
      <td>1560254416</td>
      <td>9781560254416</td>
      <td>0</td>
      <td>512</td>
      <td>156</td>
      <td>20</td>
      <td>2004</td>
      <td>Da Capo Press</td>
      <td>1</td>
      <td>4.06</td>
      <td>633.36</td>
      <td>81.20</td>
      <td>2078.72</td>
    </tr>
    <tr>
      <th>45633</th>
      <td>You Bright and Risen Angels</td>
      <td>William T. Vollmann</td>
      <td>4.08</td>
      <td>0140110879</td>
      <td>9780140110876</td>
      <td>0</td>
      <td>635</td>
      <td>783</td>
      <td>56</td>
      <td>1988</td>
      <td>Penguin Books</td>
      <td>1</td>
      <td>4.08</td>
      <td>3194.64</td>
      <td>228.48</td>
      <td>2590.80</td>
    </tr>
    <tr>
      <th>45634</th>
      <td>The Ice-Shirt (Seven Dreams #1)</td>
      <td>William T. Vollmann</td>
      <td>3.96</td>
      <td>0140131965</td>
      <td>9780140131963</td>
      <td>0</td>
      <td>415</td>
      <td>820</td>
      <td>95</td>
      <td>1993</td>
      <td>Penguin Books</td>
      <td>1</td>
      <td>3.96</td>
      <td>3247.20</td>
      <td>376.20</td>
      <td>1643.40</td>
    </tr>
    <tr>
      <th>45639</th>
      <td>Poor People</td>
      <td>William T. Vollmann</td>
      <td>3.72</td>
      <td>0060878827</td>
      <td>9780060878825</td>
      <td>0</td>
      <td>434</td>
      <td>769</td>
      <td>139</td>
      <td>2007</td>
      <td>Ecco</td>
      <td>1</td>
      <td>3.72</td>
      <td>2860.68</td>
      <td>517.08</td>
      <td>1614.48</td>
    </tr>
    <tr>
      <th>45641</th>
      <td>Las aventuras de Tom Sawyer</td>
      <td>Mark Twain</td>
      <td>3.91</td>
      <td>8497646983</td>
      <td>9788497646987</td>
      <td>2</td>
      <td>272</td>
      <td>113</td>
      <td>12</td>
      <td>2006</td>
      <td>Edimat Libros</td>
      <td>1</td>
      <td>3.91</td>
      <td>441.83</td>
      <td>46.92</td>
      <td>1063.52</td>
    </tr>
  </tbody>
</table>
<p>11042 rows × 16 columns</p>
</div>



After dropping non-numeric columns, the dataframe is ready to be fitted into the models again.


```python
df2_processed = df2.drop(columns=["title", "authors", "isbn", "isbn13", "publisher"])
```


```python
df2_processed
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>average_rating</th>
      <th>language_code</th>
      <th>num_pages</th>
      <th>ratings_count</th>
      <th>text_reviews_count</th>
      <th>publication_date</th>
      <th>occurence_count</th>
      <th>occ_weighted</th>
      <th>rating_count_weighted</th>
      <th>text_review_count_weighted</th>
      <th>page_weighted</th>
    </tr>
    <tr>
      <th>bookID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4.57</td>
      <td>0</td>
      <td>652</td>
      <td>2095690</td>
      <td>27591</td>
      <td>2006</td>
      <td>2</td>
      <td>9.14</td>
      <td>9577303.30</td>
      <td>126090.87</td>
      <td>2979.64</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.49</td>
      <td>0</td>
      <td>870</td>
      <td>2153167</td>
      <td>29221</td>
      <td>2004</td>
      <td>1</td>
      <td>4.49</td>
      <td>9667719.83</td>
      <td>131202.29</td>
      <td>3906.30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.42</td>
      <td>0</td>
      <td>352</td>
      <td>6333</td>
      <td>244</td>
      <td>2003</td>
      <td>2</td>
      <td>8.84</td>
      <td>27991.86</td>
      <td>1078.48</td>
      <td>1555.84</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.56</td>
      <td>0</td>
      <td>435</td>
      <td>2339585</td>
      <td>36325</td>
      <td>2004</td>
      <td>2</td>
      <td>9.12</td>
      <td>10668507.60</td>
      <td>165642.00</td>
      <td>1983.60</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4.78</td>
      <td>0</td>
      <td>2690</td>
      <td>41428</td>
      <td>164</td>
      <td>2004</td>
      <td>1</td>
      <td>4.78</td>
      <td>198025.84</td>
      <td>783.92</td>
      <td>12858.20</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>45631</th>
      <td>4.06</td>
      <td>0</td>
      <td>512</td>
      <td>156</td>
      <td>20</td>
      <td>2004</td>
      <td>1</td>
      <td>4.06</td>
      <td>633.36</td>
      <td>81.20</td>
      <td>2078.72</td>
    </tr>
    <tr>
      <th>45633</th>
      <td>4.08</td>
      <td>0</td>
      <td>635</td>
      <td>783</td>
      <td>56</td>
      <td>1988</td>
      <td>1</td>
      <td>4.08</td>
      <td>3194.64</td>
      <td>228.48</td>
      <td>2590.80</td>
    </tr>
    <tr>
      <th>45634</th>
      <td>3.96</td>
      <td>0</td>
      <td>415</td>
      <td>820</td>
      <td>95</td>
      <td>1993</td>
      <td>1</td>
      <td>3.96</td>
      <td>3247.20</td>
      <td>376.20</td>
      <td>1643.40</td>
    </tr>
    <tr>
      <th>45639</th>
      <td>3.72</td>
      <td>0</td>
      <td>434</td>
      <td>769</td>
      <td>139</td>
      <td>2007</td>
      <td>1</td>
      <td>3.72</td>
      <td>2860.68</td>
      <td>517.08</td>
      <td>1614.48</td>
    </tr>
    <tr>
      <th>45641</th>
      <td>3.91</td>
      <td>2</td>
      <td>272</td>
      <td>113</td>
      <td>12</td>
      <td>2006</td>
      <td>1</td>
      <td>3.91</td>
      <td>441.83</td>
      <td>46.92</td>
      <td>1063.52</td>
    </tr>
  </tbody>
</table>
<p>11042 rows × 11 columns</p>
</div>




```python
plt.figure(figsize=(10,8))
sns.heatmap(df2_processed.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.show()
```


    
![png](output_82_0.png)
    



```python
df2_train, df2_test = train_test_split(df2_processed, test_size=0.20, random_state=42)

X_train2 = df2_train.drop("average_rating", axis=1)
y_train2 = df2_train["average_rating"]

X_test2 = df2_test.drop("average_rating", axis=1)
y_test2 = df2_test["average_rating"]
```


```python
print(len(df2_processed))
print(len(X_train2))
print(len(X_test2))
```

    11042
    8833
    2209
    


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


```python
print(results2.values())
print(mean_sq_error2.values())
```

    dict_values([0.8147444153821022, 0.9940385883286225, 0.994373696209495, 0.993304644786113, 0.6050417083950768])
    dict_values([np.float64(0.15305002333345824), np.float64(0.027455054014627795), np.float64(0.02667223044417847), np.float64(0.029096085026598627), np.float64(0.2234721274812092)])
    


```python
plt.figure(figsize=(10, 6))
sns.barplot(x=list(results2.values()), y=list(results2.keys()), palette='viridis')
plt.xlabel('R2 Score')
plt.title('Regressor Model Accuracy (R2 Score) - Revised')
plt.xlim(0, 1)
plt.show()
```

    


    
![png](output_87_1.png)
    


### 6. Conclusion

As can be seen from the above plot, models performed significantly better after the addition of the new features, with three of our five models performing in the 99.33 - 99.45 range. Furthermore, the RandomForestRegressor model performed the best in this iteration with an R2 score of 99.44% (when rounded to two decimals). Therefore, we select it for our main model to be used in the final application. Finally, below we run the prediction again but this time seperately for the RandomForestRegressor model.


```python
random_forest_reg = RandomForestRegressor(random_state=42)
random_forest_reg.fit(X_train2, y_train2)
```








```python
y_predicted = np.round(random_forest_reg.predict(X_test2), 2)
```

#### Final Model Evaluation for RandomForestRegressor


```python
accuracy = (y_predicted == y_test2).sum()/len(y_test2)
print("Accuracy of RandomForestRegressor calculated by hand:", np.round((accuracy * 100), 2), "%")
```

    Accuracy of RandomForestRegressor calculated by hand: 93.39 %
    


```python
score = r2_score(y_test2, y_predicted)
print("R2 Score of RandomForestRegressor:", np.round((score * 100), 2))
```

    R2 Score of RandomForestRegressor: 99.44
    


```python
error = np.sqrt(mean_squared_error(y_test2, y_predicted))
print("MSE for RandomForestRegressor:", np.round(error, 3))
```

    MSE for RandomForestRegressor: 0.027
    
