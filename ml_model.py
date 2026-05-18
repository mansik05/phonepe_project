import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------------------
# LOAD DATA
# -----------------------------------

df = pd.read_csv("transactions.csv")

# -----------------------------------
# FEATURE ENGINEERING
# -----------------------------------

state_data = df.groupby("state")[["amount", "count"]].sum().reset_index()

print(state_data.head())

# -----------------------------------
# NORMALIZATION (VERY IMPORTANT)
# -----------------------------------

scaler = StandardScaler()

X = state_data[["amount", "count"]]
X_scaled = scaler.fit_transform(X)

# -----------------------------------
# ELBOW METHOD (to find best clusters)
# -----------------------------------

wcss = []

for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 10), wcss, marker="o")
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# -----------------------------------
# APPLY KMEANS
# -----------------------------------

kmeans = KMeans(n_clusters=3, random_state=42)
state_data["cluster"] = kmeans.fit_predict(X_scaled)

# -----------------------------------
# VISUALIZATION
# -----------------------------------

plt.figure(figsize=(10,6))

sns.scatterplot(
    x="amount",
    y="count",
    hue="cluster",
    data=state_data,
    palette="viridis",
    s=100
)

plt.title("State Segmentation using K-Means Clustering")
plt.show()

# -----------------------------------
# MODEL EVALUATION
# -----------------------------------

score = silhouette_score(X_scaled, state_data["cluster"])
print("Silhouette Score:", score)

# -----------------------------------
# INSIGHT OUTPUT
# -----------------------------------

print(state_data.sort_values("cluster"))