import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# LOAD DATA
# -----------------------------------

df = pd.read_csv("transactions.csv")

print("\nDATA PREVIEW:\n")
print(df.head())

print("\nDATASET SHAPE:\n")
print(df.shape)

# SET STYLE
sns.set_style("whitegrid")

# =========================================================
#                UNIVARIATE ANALYSIS
# =========================================================

# ---------------------------------------------------------
# 1. Histogram of Transaction Amount
# ---------------------------------------------------------

plt.figure(figsize=(12,5))

sns.histplot(
    df["amount"],
    bins=50,
    kde=True,
    color="purple"
)

plt.title("Distribution of Transaction Amount")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 2. Boxplot of Transaction Amount
# ---------------------------------------------------------

plt.figure(figsize=(10,5))

sns.boxplot(
    x=df["amount"],
    color="orange"
)

plt.title("Boxplot of Transaction Amount")
plt.xlabel("Transaction Amount")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 3. Pie Chart of Transaction Types
# ---------------------------------------------------------

type_sum = df.groupby("type")["amount"].sum()

plt.figure(figsize=(8,8))

plt.pie(
    type_sum,
    labels=type_sum.index,
    autopct="%1.1f%%",
    startangle=140
)

plt.title("Transaction Type Share")

plt.show()

# ---------------------------------------------------------
# 4. Countplot of Years
# ---------------------------------------------------------

plt.figure(figsize=(8,5))

sns.countplot(
    x="year",
    data=df,
    palette="Set2"
)

plt.title("Transaction Records by Year")
plt.xlabel("Year")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 5. Distribution of Transaction Counts
# ---------------------------------------------------------

plt.figure(figsize=(12,5))

sns.histplot(
    df["count"],
    bins=40,
    kde=True,
    color="green"
)

plt.title("Distribution of Transaction Counts")
plt.xlabel("Transaction Count")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# =========================================================
#                BIVARIATE ANALYSIS
# =========================================================

# ---------------------------------------------------------
# 6. State vs Transaction Amount
# ---------------------------------------------------------

state_data = df.groupby("state")["amount"].sum().reset_index()

plt.figure(figsize=(16,7))

sns.barplot(
    x="state",
    y="amount",
    data=state_data,
    palette="viridis"
)

plt.xticks(rotation=90, fontsize=8)

plt.title("Total Transaction Amount by State")
plt.xlabel("States")
plt.ylabel("Transaction Amount")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 7. Transaction Type vs Amount
# ---------------------------------------------------------

type_data = df.groupby("type")["amount"].sum().reset_index()

plt.figure(figsize=(12,6))

sns.barplot(
    x="type",
    y="amount",
    data=type_data,
    palette="Set2"
)

plt.xticks(rotation=20)

plt.title("Transaction Type Distribution")
plt.xlabel("Transaction Type")
plt.ylabel("Transaction Amount")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 8. Year vs Transaction Amount
# ---------------------------------------------------------

year_data = df.groupby("year")["amount"].sum().reset_index()

plt.figure(figsize=(10,5))

sns.lineplot(
    x="year",
    y="amount",
    data=year_data,
    marker="o",
    linewidth=3,
    color="red"
)

plt.title("Yearly Transaction Growth")
plt.xlabel("Year")
plt.ylabel("Transaction Amount")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 9. Quarter vs Transaction Amount
# ---------------------------------------------------------

quarter_data = df.groupby("quarter")["amount"].sum().reset_index()

plt.figure(figsize=(8,5))

sns.lineplot(
    x="quarter",
    y="amount",
    data=quarter_data,
    marker="o",
    linewidth=3,
    color="blue"
)

plt.title("Quarter-wise Transaction Trend")
plt.xlabel("Quarter")
plt.ylabel("Transaction Amount")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 10. Top 10 States by Amount
# ---------------------------------------------------------

top_states = (
    df.groupby("state")["amount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

plt.figure(figsize=(12,6))

sns.barplot(
    x="state",
    y="amount",
    data=top_states,
    palette="magma"
)

plt.xticks(rotation=45)

plt.title("Top 10 States by Transaction Amount")
plt.xlabel("State")
plt.ylabel("Transaction Amount")

plt.tight_layout()
plt.show()

# =========================================================
#              MULTIVARIATE ANALYSIS
# =========================================================

# ---------------------------------------------------------
# 11. Correlation Heatmap
# ---------------------------------------------------------

plt.figure(figsize=(6,5))

sns.heatmap(
    df[["count", "amount"]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Count and Amount")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 12. Scatterplot (Count vs Amount by Year)
# ---------------------------------------------------------

plt.figure(figsize=(10,6))

sns.scatterplot(
    x="count",
    y="amount",
    hue="year",
    data=df
)

plt.title("Transaction Count vs Amount by Year")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 13. Pairplot
# ---------------------------------------------------------

sns.pairplot(
    df[["count", "amount", "year", "quarter"]],
    diag_kind="kde"
)

plt.show()

# ---------------------------------------------------------
# 14. Heatmap (Year vs Quarter)
# ---------------------------------------------------------

pivot_table = df.pivot_table(
    values="amount",
    index="year",
    columns="quarter",
    aggfunc="sum"
)

plt.figure(figsize=(8,5))

sns.heatmap(
    pivot_table,
    annot=True,
    fmt=".0f",
    cmap="YlGnBu"
)

plt.title("Year vs Quarter Transaction Heatmap")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 15. Multi-line Plot (Type over Years)
# ---------------------------------------------------------

multi_data = df.groupby(
    ["year", "type"]
)["amount"].sum().reset_index()

plt.figure(figsize=(12,6))

sns.lineplot(
    x="year",
    y="amount",
    hue="type",
    data=multi_data,
    marker="o"
)

plt.title("Transaction Types Over Years")
plt.xlabel("Year")
plt.ylabel("Transaction Amount")

plt.tight_layout()
plt.show()

print("\nALL 15 UBM CHARTS GENERATED SUCCESSFULLY!\n")