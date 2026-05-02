import pandas as pd
import matplotlib.pyplot as plt
from main import run_experiment   

df = pd.read_csv("synthetic_heart_disease_dataset.csv")

sizes = [100, 500, 1000, 2000, 10000, 20000, 30000, 40000]
knn_times = []
dt_times = []

for size in sizes:
    sample_df = df.sample(n=size, random_state=42).copy()

    knn_time, dt_time = run_experiment(sample_df)

    knn_times.append(knn_time)
    dt_times.append(dt_time)

# Plot graph
plt.figure()
plt.plot(sizes, knn_times, marker='o', label="KNN")
plt.plot(sizes, dt_times, marker='o', label="Decision Tree")
plt.xlabel("Dataset Size")
plt.ylabel("Time")
plt.title("KNN vs Decision Tree Performance")
plt.legend()
plt.grid()
plt.show()