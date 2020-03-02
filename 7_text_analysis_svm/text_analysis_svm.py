import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.feature_extraction.text import TﬁdfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


newsgroups=datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
y=newsgroups.target

#preprocessing data to TF-IDF
vectorizer=TﬁdfVectorizer()
x_train=vectorizer.fit_transform(newsgroups.data)
x_test=vectorizer.transform(newsgroups.data)

#preparing best C
grid={'C': np.power(10.0, np.arange(-5, 6))}
kf=KFold(n_splits=5, shuffle=True, random_state=241)
clf=SVC(kernel='linear', random_state=241)
gs=GridSearchCV(clf, grid, scoring='accuracy', cv=kf)
gs.fit(x_train,y)
best_c=gs.best_estimator_.C

#new fitting to get 10 best words
clf_new=SVC(C=best_c, kernel='linear', random_state=241)
clf_new.fit(x_train, y)
df2 = pd.DataFrame( np.transpose(clf_new.coef_.toarray()), index=np.asarray(vectorizer.get_feature_names()) , columns=["col1"])
df2['col1']=df2['col1'].abs()
df2=df2.sort_values(by='col1', ascending=False)
print(df2.head(n=10))