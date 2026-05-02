import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#Load dataset
df = pd.read_csv("synthetic_heart_disease_dataset.csv")
#print(df.head())

# Encoding
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Smoking"] = df["Smoking"].map({"Never": 0, "Former": 1, "Current": 2})
df["Physical_Activity"] = df["Physical_Activity"].map({"Sedentary": 0, "Moderate": 1, "Active": 2})
df["Diet"] = df["Diet"].map({"Unhealthy": 0, "Average": 1, "Healthy": 2})
df["Stress_Level"] = df["Stress_Level"].map({"Low": 0, "Medium": 1, "High": 2})

#Split up the featuers and target
X = df.drop(["Heart_Disease", "Alcohol_Intake"], axis=1)
y = df["Heart_Disease"]

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (required step for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Algorithm 1: K-Nearest Neighbour Model

knn = KNeighborsClassifier(n_neighbors=5)

startknn = time.time()

#Train & Predict model
knn.fit(X_train, y_train)
predknn = knn.predict(X_test)

endknn = time.time()

#Compute Results
knn_time = (endknn - startknn) * 1000  
knn_acc = accuracy_score(y_test, predknn)

# Algorithm 2:  Decision Tree Model

dt = DecisionTreeClassifier(max_depth=3, random_state=42)

startdt = time.time()

#Train & Predict
dt.fit(X_train, y_train)
preddt = dt.predict(X_test)

enddt = time.time()

#Compute Results
dt_time = (enddt - startdt) * 1000  
dt_acc = accuracy_score(y_test, preddt)  

#Final Results for both Algorithms

print("\nFinal Results: ")

print("\nKNN Results:")
print("Accuracy:", knn_acc)
print("Time:", knn_time)

print("\nDecision Tree Results:")
print("Accuracy:", dt_acc)
print("Time:", dt_time)

# Function for Graph File 
def run_experiment(df):
    import time
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    # Encoding
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    df["Smoking"] = df["Smoking"].map({"Never": 0, "Former": 1, "Current": 2})
    df["Physical_Activity"] = df["Physical_Activity"].map({"Sedentary": 0, "Moderate": 1, "Active": 2})
    df["Diet"] = df["Diet"].map({"Unhealthy": 0, "Average": 1, "Healthy": 2})
    df["Stress_Level"] = df["Stress_Level"].map({"Low": 0, "Medium": 1, "High": 2})

    # Split features and target
    X = df.drop(["Heart_Disease", "Alcohol_Intake"], axis=1)
    y = df["Heart_Disease"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    startknn = time.time()
    knn.fit(X_train, y_train)
    knn.predict(X_test)
    knn_time = (time.time() - startknn) * 1000

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    startdt = time.time()
    dt.fit(X_train, y_train)
    dt.predict(X_test)
    dt_time = (time.time() - startdt) * 1000

    return knn_time, dt_time