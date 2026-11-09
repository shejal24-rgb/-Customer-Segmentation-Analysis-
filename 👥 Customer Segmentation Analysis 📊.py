import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

file_path = Path(__file__).parent / "ProjectDA" / "customer_segmentation.csv"

df = pd.read_csv( "customer_segmentation.csv")

print(df.head())
print(df.info())
print(df.describe())

print("Total Customers:", df["Customer_ID"].nunique())
print("Average Age:", np.mean(df["Age"]))
print("Average Annual Income:", np.mean(df["Annual_Income"]))
print("Average Spending Score:", np.mean(df["Spending_Score"]))
print("Total Purchases:", df["Purchase_Frequency"].sum())

income_by_city = df.groupby("City")["Annual_Income"].mean()
print(income_by_city)

spending_by_gender = df.groupby("Gender")["Spending_Score"].mean()
print(spending_by_gender)

df["Customer_Segment"] = np.where(
    df["Spending_Score"] >= 70,
    "High Spenders",
    np.where(df["Spending_Score"] >= 40, "Medium Spenders", "Low Spenders")
)

print(df[["Customer_ID", "Customer_Segment"]])

plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=8, kde=True)
plt.title("Customer Age Distribution")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Annual_Income", y="Spending_Score", hue="Gender")
plt.title("Annual Income vs Spending Score")
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="City", y="Annual_Income")
plt.title("Average Annual Income by City")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Customer_Segment")
plt.title("Customer Segmentation")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Gender", y="Spending_Score")
plt.title("Spending Score by Gender")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Age", y="Purchase_Frequency", hue="Customer_Segment")
plt.title("Age vs Purchase Frequency")
plt.show()