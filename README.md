
# Simple SMS Spam Classifier

Spam Classification is one of the known problems in todays time. In this project we applied supervised learning classification algorithms. We have used https://archive.ics.uci.edu/ml/datasets/sms+spam+collection public dataset. The dataset contains two features message and label. Message feature refers to SMS and label feature refers to either spam or ham for each SMS.

| Features | Descrption |
|----------|------------|
|Messages  |SMS message |
|Label     |Ham or Spam |

Before building any model we can see that dataset is imbalanced, so we went for F1 Score to measure performance. We apply data cleaning operations to get clean data.

After obtaining the cleaned dataset, we created lemmas of SMS corpus separately by using NLTK, we removed the punctuations and stopwords as they do not contain anything meaningful and then, we generated TF-IDF of SMS corpus, respectively.

We then build models using Logistic Regression, Naive Bayes Classifier, Support Vector Classifier, Gradient Boosting Classifier. With parameter tuning we finalized Support Vector Classifier model.

| Model | F1 Score Result |
|-------|--------|
|Logistic Regression|86.7%|
|Naive Bayes Classifier|91.4%|
|Support Vector Classifier|92.8%|
|Gradient Boosting Classifier|88.02%|
|Support Vector Classifier (with **GridSearchCV**)|93.3%|


## Libraires used

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
```

  
