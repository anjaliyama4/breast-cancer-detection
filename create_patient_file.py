from sklearn.datasets import load_breast_cancer
import pandas as pd
import os

# Load Wisconsin Breast Cancer Dataset
data = load_breast_cancer()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Create folder if it doesn't exist
if not os.path.exists("sample_patients"):
    os.mkdir("sample_patients")

# Create first 10 patient CSV files
for i in range(10):
    patient = df.iloc[[i]]  # One patient only
    filename = f"sample_patients/patient_{i+1:03}.csv"
    patient.to_csv(filename, index=False)

print("===================================")
print("10 Patient CSV files created!")
print("Location: sample_patients/")
print("===================================")