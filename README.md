# Heart Disease Prediction using K-Nearest Neighbors (KNN)

## Overview

This project applies **Machine Learning** to predict whether a patient has **heart disease** based on clinical and medical attributes.
The model uses the **K-Nearest Neighbors (KNN)** algorithm for classification.

The workflow includes data exploration, preprocessing, model training, and evaluation using standard machine learning techniques.

---

## Dataset

The dataset contains **1025 patient records** and **13 medical features**.

Some of the features include:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol Level
* Maximum Heart Rate
* Exercise Induced Angina
* ST Depression
* Number of Major Vessels

### Target Variable

* **0 → No Heart Disease**
* **1 → Heart Disease**

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

---

## Machine Learning Pipeline

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Feature–Target Separation
4. Train-Test Split
5. Feature Scaling (StandardScaler)
6. K-Nearest Neighbors Model Training
7. Model Prediction
8. Model Evaluation (Accuracy, Confusion Matrix, Classification Report)
9. Data Visualization

---

## Model Evaluation

The model performance is evaluated using:

* **Accuracy Score**
* **Confusion Matrix**
* **Precision**
* **Recall**
* **F1-Score**

Example confusion matrix visualization:

![Confusion Matrix](images/confusion_matrix.png)

---

## Visualizations

### Cholesterol Distribution

![Cholesterol Histogram](images/cholesterol_histogram.png)

### Heart Disease Distribution

![Target Distribution](images/target_distribution.png)

---

## Installation

Install the required dependencies:

```
pip install -r requirements.txt
```

---

## Run the Project

Execute the Python script:

```
python main.py
```

---

## Project Structure

```
ml-heart-dis/
│
├── data/
│   └── heart.csv
│
├── main.py
├── README.md
└── requirements.txt
```

This project demonstrates the application of **K-Nearest Neighbors (KNN)** for medical data classification and highlights the full machine learning workflow from data analysis to model evaluation.

