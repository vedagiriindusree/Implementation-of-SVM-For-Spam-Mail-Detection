# Implementation-of-SVM-For-Spam-Mail-Detection
### NAME : VEDAGIRI INDUSREE
### REG NO : 212223230236
## AIM:

To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: Start the program.

Step 2: Import the required packages.

Step 3: Import the dataset to operate on.

Step 4: Split the dataset.

Step 5: Predict the required output.

Step 6: End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: VEDAGIRI INDUSREE
RegisterNumber: 212223230236
*/

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')


data.head()


data.info()


data.isnull().sum()


x=data["v1"].values
y=data["v2"].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)


from sklearn.feature_extraction.text import CountVectorizer
#countvectorizer is a method to convert text to numerical data. The text is transformed to a sparse matrix
cv=CountVectorizer()


x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)


from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy


from sklearn.metrics import confusion_matrix,classification_report
con=confusion_matrix(y_test,y_pred)
print(con)
cl=classification_report(y_test,y_pred)
print(cl)

```

## Output:

## data.head:

![image](https://github.com/user-attachments/assets/19e5bc1c-57f4-4e4e-a701-a886e558b005)

## data.info:

![image](https://github.com/user-attachments/assets/777a0e88-d6bc-40d8-b435-cb5cbf8d7c5e)

## data.isnull:

![image](https://github.com/user-attachments/assets/ffae58fd-cfee-4e7d-9e4a-f38aed4aff4d)

## accuracy :

![image](https://github.com/user-attachments/assets/553427f9-b45b-430b-a06d-fde95dd90e4c)

## confusion matrix:

![image](https://github.com/user-attachments/assets/bbd88924-18ce-4bfe-8dd6-6298164aab92)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
