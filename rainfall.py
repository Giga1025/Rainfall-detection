import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics
from sklearn.metrics import r2_score

path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv'
df = pd.read_csv(path)
df.columns

df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
df_sydney_processed.columns

df_sydney_processed.replace(['No', 'Yes'], [0,1], inplace=True)
df_sydney_processed.drop('Date',axis=1,inplace=True)
df_sydney_processed = df_sydney_processed.astype(float)

features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
Y = df_sydney_processed['RainTomorrow']

x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size = 0.2, random_state = 10)

LinearReg = LinearRegression()
LinearReg.fit(x_train,y_train)
print("coefficient: ", LinearReg.coef_)
print("Intercept: ", LinearReg.intercept_)

predictions = LinearReg.predict(x_test)
LinearRegression_MAE = np.mean(np.abs(predictions - y_test))
LinearRegression_MSE = np.mean((predictions - y_test)**2)
LinearRegression_R2 =  r2_score(y_test, predictions)
Report = pd.DataFrame({'MAE':[LinearRegression_MAE], 'MSE':[LinearRegression_MSE], 'R2':[LinearRegression_R2]})

K = 4
KNN =KNeighborsClassifier(n_neighbors = K).fit(x_train, y_train)
KNN 

predictions = KNN.predict(x_test)
print("Train set Accuracy: ", metrics.accuracy_score(y_train, KNN.predict(x_train)))
KNN_Accuracy_Score = accuracy_score(y_test, predictions)
KNN_JaccardIndex = jaccard_score(y_test, predictions)
KNN_F1_Score = f1_score(y_test, predictions)

Report = pd.DataFrame({'Accuracy_score':[KNN_Accuracy_Score], 'JaccardIndex':[KNN_JaccardIndex], 'KNN_F1_Score':[KNN_F1_Score]})


Tree = DecisionTreeClassifier(criterion="entropy", max_depth=4).fit(x_train,y_train)
predictions = Tree.predict(x_test)
print('Accuracy score for testing set: ', accuracy_score(y_test, predictions ))
Tree_Accuracy_Score = accuracy_score(y_test, predictions)
Tree_JaccardIndex = jaccard_score(y_test, predictions)
Tree_F1_Score = f1_score(y_test, predictions)
Report = pd.DataFrame({'Accuracy_score':[Tree_Accuracy_Score], 'JaccardIndex':[Tree_JaccardIndex], 'Tree_F1_Score':[Tree_F1_Score]})
Report

x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=1)

LR = LogisticRegression(C = 0.01, solver= "liblinear").fit(x_train, y_train)
predictions = LR.predict(x_test)
predict_proba = LR.predict_proba(x_test)
LR_Accuracy_Score = accuracy_score(y_test,predictions)
LR_JaccardIndex = jaccard_score(y_test, predictions)
LR_F1_Score = f1_score(y_test, predictions)
LR_Log_Loss = log_loss(y_test, predict_proba)
Report = pd.DataFrame({'Accuracy_score':[LR_Accuracy_Score], 'JaccardIndex':[LR_JaccardIndex], 'LR_F1_Score':[LR_F1_Score], 'Logloss':[LR_Log_Loss]})



SVM = svm.SVC(kernel="linear", C = 0.01,gamma='scale', random_state=1).fit(x_train, y_train)
predictions = SVM.predict(x_test)
SVM_Accuracy_Score = accuracy_score(y_test,predictions)
SVM_JaccardIndex = jaccard_score(y_test, predictions)
SVM_F1_Score = f1_score(y_test, predictions,average='weighted')
Report = pd.DataFrame({'Accuracy_score':[SVM_Accuracy_Score], 'JaccardIndex':[SVM_JaccardIndex], 'SVM_F1_Score':[SVM_F1_Score]})
Report



Reportf= pd.DataFrame({"Accuracy_score": [KNN_Accuracy_Score, Tree_Accuracy_Score, LR_Accuracy_Score, SVM_Accuracy_Score],
                        "Jaccard_index":[KNN_JaccardIndex,Tree_JaccardIndex,LR_JaccardIndex,SVM_JaccardIndex],
                        "F1_Score":[KNN_F1_Score,Tree_F1_Score, LR_F1_Score, SVM_F1_Score ]
                        },index=["KNN", "Decision Tree", "Logistic Regression", "SVM"])
Reportf.loc["Logistic Regression", "Log_Loss"] = LR_Log_Loss
print(Reportf)

