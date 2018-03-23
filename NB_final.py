import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

'''
1.Prepare data
'''
raw = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)

# split data to training set and testing set
def review_series_to_list(review_series):
    review_list=[]
    n_review = len(review_series)
    for i in range(0,n_review):
        review_list.append(review_series[i])
    return review_list  

train_review_list = review_series_to_list(raw['review'])

X_train, X_test, y_train, y_test = train_test_split(
    train_review_list, raw['sentiment'], test_size=0.33, random_state=42)

'''
2.Train model---Naive Bayes
'''
nb_model = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])

nb_fit = nb_model.fit(X_train, y_train)

###Prediction and evaluation
nb_predicted = nb_model.predict(X_test)
nb_accuracy = np.mean(predicted == y_test) 
print (nb_accuracy)
# 0.86484848484848487

'''
3.Train model---SVM
'''
SVM_model = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, n_iter=5, random_state=42)),
])
SVM_model.fit(X_train, y_train)  

svm_predicted = SVM_model.predict(X_test)
svm_accuracy = np.mean(predicted == y_test)  
print(svm_accuracy)          
#0.85248484848484851

'''
4.Performance
'''
print(metrics.classification_report(y_test, nb_predicted,
    target_names=['Rating < 5(0)','Rating >=7(1)']))
metrics.confusion_matrix(y_test, nb_predicted)

print(metrics.classification_report(y_test, svm_predicted,
    target_names=['Rating < 5(0)','Rating >=7(1)']))
metrics.confusion_matrix(y_test, svm_predicted)

'''
5.Tuning parameter by using grid search
'''