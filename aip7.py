import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

warnings.filterwarnings("ignore")

data = pd.read_csv("apple_and_oranges.csv")
print(data)

#test_size = 0.2 => 20% test , 80% training
training_set, test_set = train_test_split(data, test_size = 0.2, random_state = 1)
X_train = training_set.iloc[:,0:2].values
Y_train = training_set.iloc[:,2].values
X_test = test_set.iloc[:,0:2].values
Y_test = test_set.iloc[:,2].values

#base_estimator: It is weak learner used to train thje model
#it uses DecisionTreeClassifier as default weak learning for training purpose
#you can also specify diffrent different machine learning algorithms

adaboost = AdaBoostClassifier(n_estimators = 100, base_estimator=None, learning_rate = 1, random_state=1)
adaboost.fit(X_train, Y_train)
Y_pred = adaboost.predict(X_test)
test_set["Predictions"] = Y_pred
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
results = confusion_matrix(Y_test, Y_pred)
print("Confusion matrix\n" , results)
print("Accuracy score: " ,accuracy_score(Y_test, Y_pred ))
print("Report: ")
print(classification_report(Y_test, Y_pred))
