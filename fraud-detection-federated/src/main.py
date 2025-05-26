import pandas as pd

# Load the synthetic transaction dataset
df = pd.read_csv("../data/synthetic_data.csv")

# Basic fraud statistics
fraud_count = df["is_fraud"].sum()
total = len(df)
fraud_rate = fraud_count / total * 100

print(f"Total Transactions: {total}")
print(f"Fraudulent Transactions: {fraud_count}")
print(f"Fraud Rate: {fraud_rate:.2f}%")
