
# Restaurant Review Classifier
This is a simple NLP project based on the [NLP section of A-Z Machine Learning Course on Udemy](https://www.udemy.com/machinelearning/learn/v4/t/lecture/6085634?start=0)

The objective of this exercise is to identify the best model for classifying the review comments of a restaurant. We clean the dataset and make vectors out of them according to the bag of words model.

## Index
#### 1. [Preprocessing](#preprocessing)
#### 2. [Gaussian Naive Bayes](#gnb)
#### 3. [Decision Tree Classifier](#dtc)
#### 4. [Random Forest Classifier](#RFC)
#### 5. [Predictor Function](#predictor)
#### 6. [Conclusion](#conclusion)

<a id='preprocessing'></a>
### Preprocessing

##### Dataset

<div>
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

##### steps taken
- Removal of punctuations and symbols
- Removing the stop words
- Tokenizing after stemming the different words.
- Building the vectors from the induvidual reviews.



<a id='gnb'></a>
## Summary of  Gaussian Naive Bayes  ##

##### The confusion matrix is :

\begin{bmatrix}
    55 & 42 \\
    12 & 91
\end{bmatrix}

$ accuracy =  0.73$

$precision =  0.567010309278$

$recall =  0.820895522388$

$F1score =  0.670731707317$


<a id='dtc'></a>
## Summary of  Decision Tree Classifier  ##

##### The confusion matrix is :

\begin{bmatrix}
    74 & 23 \\
    35 & 68
\end{bmatrix}

$ accuracy =  0.71$

$precision =  0.762886597938$

$recall =  0.678899082569$

$F1score =  0.718446601942$

<a id='RFC'></a>
## Summary of  Random Forest Classifier  ##

##### The confusion matrix is :

\begin{bmatrix}
    87 & 10 \\
    46 & 57
\end{bmatrix}

$ accuracy =  0.72$

$precision =  0.896907216495$

$recall =  0.654135338346$

$F1score =  0.75652173913$

<a id='predictor'></a>
### Predictor

A sample predictor was created for implementing in our django app. The basic logic is to classify the comment with all the three models that we tried and then using the average of the result in order to predict the final result. This predictor takes the input in the form of a string.

## Summary of  Predictor function  ##

##### The confusion matrix is :

\begin{bmatrix}
    85 & 15 \\
    32 & 71
\end{bmatrix}

$ accuracy =  0.765$

$precision =  0.845360824742$

$recall =  0.719298245614$

$F1score =  0.777251184834$

##### Pickling for use in our Django project

The three trained models were pickled using python's pickle library and then used inside the Django project.

<a id='conclusion'></a>
### Conclusion 

In conclusion, we can say that none of these methods do a perfect job in classifying the reviews perfectly. However we can say that the best result was obtained for Random Forest Classifier. And even better result was obtained from the predictor function which aggregates the three classifiers. Another one factor we need to consider is that this model was built on only very limited dataset and has its limitations. Altogether we are able to get fairly good results for a basic implementatio on a web 
