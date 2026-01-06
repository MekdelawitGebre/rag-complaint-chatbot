# src/task1_eda_preprocessing.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# -------------------------------
# 1. Load the dataset
# -------------------------------
csv_path = "data/raw/complaints.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} not found. Please check your path.")

# Read CSV safely
df = pd.read_csv(csv_path, low_memory=False)

# Normalize column names: lowercase + underscores + strip spaces
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print("Columns after normalization:", df.columns.tolist())

# -------------------------------
# 2. Basic EDA
# -------------------------------
print(f"Total complaints: {len(df)}")
print("Sample rows:")
print(df.head())

# Distribution across products
plt.figure(figsize=(10,5))
sns.countplot(data=df, x='product', order=df['product'].value_counts().index)
plt.xticks(rotation=45)
plt.title("Complaints by Product")
plt.tight_layout()
plt.savefig("data/processed/complaints_by_product.png")
plt.show()

# Length of narratives
df['narrative_length'] = df['consumer_complaint_narrative'].dropna().apply(lambda x: len(str(x).split()))
plt.figure(figsize=(10,5))
sns.histplot(df['narrative_length'], bins=50)
plt.title("Distribution of Complaint Narrative Lengths (words)")
plt.xlabel("Word count")
plt.ylabel("Number of complaints")
plt.tight_layout()
plt.savefig("data/processed/narrative_length_distribution.png")
plt.show()

# -------------------------------
# 3. Filter for products of interest
# -------------------------------
products_of_interest = ["Credit card", "Personal loan", "Savings account", "Money transfer", "Money transfers"]
df = df[df['product'].isin(products_of_interest)]
print(f"Filtered complaints: {len(df)}")

# -------------------------------
# 4. Drop rows with empty narratives
# -------------------------------
df = df[df['consumer_complaint_narrative'].notna()]
print(f"After removing empty narratives: {len(df)}")

# -------------------------------
# 5. Basic text cleaning
# -------------------------------
def clean_text(text):
    text = str(text).lower()  # lowercase
    text = re.sub(r'\s+', ' ', text)  # remove extra whitespace
    text = re.sub(r'[^a-z0-9,.!? ]', '', text)  # remove special characters
    text = text.strip()
    return text

df['cleaned_narrative'] = df['consumer_complaint_narrative'].apply(clean_text)

# -------------------------------
# 6. Save cleaned CSV
# -------------------------------
os.makedirs("data/processed", exist_ok=True)
output_path = "data/processed/filtered_complaints.csv"
df.to_csv(output_path, index=False)
print(f"Cleaned dataset saved to {output_path}")
