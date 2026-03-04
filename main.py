import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report

df=pd.read_csv("data/heart.csv")
print(df.head())
print(df.info())
print(df.describe())

df['chol'].plot(kind='hist')

plt.title('Cholesterol distribution')
plt.xlabel('Cholesterol')
plt.ylabel('Frequency')
plt.show()

print(df['sex'].value_counts())

#1. Feature–target separation
X=df.drop('target',axis=1) #feature matrix
y=df['target'] #target vector

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#feature scaling FOR KNN
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

#Prediction 
y_pred = knn.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease','Disease'],
            yticklabels=['No Disease','Disease'])

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

print(classification_report(y_test, y_pred))

#plotting
df['target'].value_counts().plot(kind= 'bar')

plt.title("Heart Disease Distribution")
plt.xlabel("Target ")
plt.ylabel("Number of Patients")
plt.show()