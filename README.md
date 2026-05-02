# Heart Disease Prediction Project

## Overview
This project compares two machine learning algorithms:
- K-Nearest Neighbors (KNN)
- Decision Tree

The goal is to analyze their performance in terms of:
- Accuracy
- Time Complexity
- Scalability

---

## Algorithms Used

### K-Nearest Neighbors (KNN)
- Type: Lazy Learning (Slower)
- Training Complexity: O(n·d) (data storage)
- Prediction Complexity: O(n·d)
- High accuracy but slow for large datasets

---

### Decision Tree
- Type: Eager Learning (Faster)
- Training Complexity: O(n log n)
- Prediction Complexity: O(log n)
- Fast prediction and scalable

---

## Technologies & Libraries
- Python
- pandas
- scikit-learn
- matplotlib

---

## Data Preprocessing
- Encoding categorical features
- Feature scaling using StandardScaler
- Removing unnecessary feature (Alcohol_Intake)

---

## Experiment Setup
We tested different dataset sizes to analyze performance:

- Small sizes: 100, 500, 1000, 2000  
- Large sizes: 10000, 20000, 30000, 40000  

---

## Results Summary
- KNN achieved higher accuracy
- Decision Tree was significantly faster
- Trade-off observed between accuracy and efficiency

## KNN provides better accuracy but is computationally expensive.  
## Decision Tree is faster and more scalable for large datasets.
