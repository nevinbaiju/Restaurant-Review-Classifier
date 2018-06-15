
# Restaurant Review Classifier
This is a simple NLP project based on the [NLP section of A-Z Machine Learning Course on Udemy](https://www.udemy.com/machinelearning/learn/v4/t/lecture/6085634?start=0)

The objective of this exercise is to identify the best model for classifying the review comments of a restaurant. We clean the dataset and make vectors out of them according to the bag of words model.

## Index
#### 1. [Preprocessing](#preprocessing)
#### 2. [Helper Functions](#hf)
#### 3. [Gaussian Naive Bayes](#gnb)

<a id='preprocessing'></a>
### Preprocessing

##### steps taken
- Removal of punctuations and symbols
- Removing the stop words
- Tokenizing after stemming the different words.
- Building the vectors from the induvidual reviews.


```python
# importing some basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split 
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
```


```python
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
dataset.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Review</th>
      <th>Liked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wow... Loved this place.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Crust is not good.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Not tasty and the texture was just nasty.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Stopped by during the late May bank holiday of...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The selection on the menu was great and so wer...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



The dataset contains the review string followed by a binary flag indicating wheather the user liked it or not.


```python
# This function will remove the unnecessary symbols, stopwords, and stem the words to tokens.
def clean_string(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    return review
```


```python
corpus = dataset['Review'].apply(clean_string)
```


```python
cv = CountVectorizer(max_features = 1500)
```


```python
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
```


```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
```

<a id='hf'></a>
### Helper Functions


```python
from sklearn.metrics import confusion_matrix
```


```python
def describe_performance(model_name, y_train, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[0][0]
    FP = cm[0][1]
    TN = cm[1][1]
    FN = cm[1][0]
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = (2*precision*recall)/(precision+recall)
    print('## Summary of ', model_name,' ##')
    print('accuracy is ', accuracy)
    print('precision is ', precision)
    print('recall is ', recall)
    print('F1 Score is ', f1)
```

<a id='gnb'></a>
### Gaussian Naive Bayes


```python
from sklearn.naive_bayes import GaussianNB
```


```python
classifier = GaussianNB()
classifier.fit(x_train, y_train)
```




    GaussianNB(priors=None)




```python
y_pred = classifier.predict(x_test)
```


```python
describe_performance('Gaussian Naive Bayes', y_train, y_pred)
```

    ## Summary of  Gaussian Naive Bayes  ##
    accuracy is  0.73
    precision is  0.567010309278
    recall is  0.820895522388
    F1 Score is  0.670731707317

