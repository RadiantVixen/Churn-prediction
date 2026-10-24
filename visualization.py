import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load dataset (replace 'data.csv' with your file path)
df = pd.read_csv("p01_bank_data/bank_data_train.csv")

# Set aesthetic style
sns.set_theme(style="whitegrid")
fig_size = (12, 6)

# 1. Target Distribution Plot
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="TARGET", palette="Set2")
plt.title("Class Distribution of TARGET")
plt.xlabel("Target Class")
plt.ylabel("Customer Count")
plt.savefig("target_distribution.png", bbox_inches="tight")
plt.close()

# 2. Numerical Comparison: Salary vs Age grouped by Target
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(
    data=df,
    x="TARGET",
    y="CLNT_SALARY_VALUE",
    ax=axes[0],
    palette="Pastel1",
    showfliers=False,
)

axes[0].set_title("Salary Distribution by Target")

sns.boxplot(
    data=df,
    x="TARGET",
    y="AGE",
    ax=axes[1],
    palette="Pastel1",
    showfliers=False,
)
axes[1].set_title("Age Distribution by Target")

plt.savefig("numerical_features_boxplot.png", bbox_inches="tight")
plt.close()

# 3. Correlation Heatmap of Key Financial & Transaction Features
key_numeric = [
    "AGE",
    "CLNT_SALARY_VALUE",
    "REST_AVG_CUR",
    "TURNOVER_PAYM",
    "CR_PROD_CNT_TOVR",
    "TARGET",
]

plt.figure(figsize=(8, 6))
corr = df[key_numeric].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap of Key Features")
plt.savefig("correlation_heatmap.png", bbox_inches="tight")
plt.close()

print(
    "Visualizations generated and saved successfully: target_distribution.png, numerical_features_boxplot.png, correlation_heatmap.png"
)

