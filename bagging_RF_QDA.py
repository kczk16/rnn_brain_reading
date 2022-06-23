import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from all_labels_download import download_data_as_dataframe
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA


# path = ...
df1 = download_data_as_dataframe(path, set_no='both', subject=1)
df2 = download_data_as_dataframe(path, set_no='side', subject=1)
df = pd.concat([df1, df2], axis=0)

values = df.values.astype('int32')
np.random.shuffle(values)
max_row = df.shape[0]
max_row -= max_row % -100

X, y = values[:, :-1], values[:, -1]

comp = 0.99
pca = PCA(n_components=comp)
X = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print("shapes before reshape")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
max_row_train = X_train.shape[0]
max_row_test = X_test.shape[0]


model = BaggingClassifier(n_estimators=50, max_features=20)
model.fit(X_train, y_train)
model_pred_train = model.predict(X_train)
print('Training accuracy (bagging) =>', accuracy_score(y_train, model_pred_train))

model_pred_test = model.predict(X_test)
print('Test accuracy (bagging) =>', accuracy_score(y_test, model_pred_test))

print("###################")

clf = QuadraticDiscriminantAnalysis(reg_param=0.7)
clf.fit(X_train, y_train)
model_pred_train = clf.predict(X_train)
print('Training accuracy (QDA) =>', accuracy_score(y_train, model_pred_train))

model_pred_test = clf.predict(X_test)
print('Test accuracy (QDA) =>', accuracy_score(y_test, model_pred_test))

print("###################")

clf = RandomForestClassifier(criterion='entropy', max_depth=30, max_features='sqrt', min_samples_leaf=1,
                             min_samples_split=2, n_estimators=250)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

print('Training accuracy (RF) =>', clf.score(X_train, y_train))
print('Test accuracy (RF) =>', clf.score(X_test, y_test))
