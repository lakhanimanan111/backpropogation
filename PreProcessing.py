import sys, math
import pandas as pd
import numpy as np

# python PreProcessing.py "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data" "C://Users/manan/abc"

# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
url = sys.argv[1] if len(sys.argv) > 1 else "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
input_dataset = pd.read_csv(url, header=None)

# Removing null or missing values
input_dataset = input_dataset.dropna()
# Converting categorical or nominal value to numerical values
for column in input_dataset.columns:
    if input_dataset[column].dtype != np.number:
         input_dataset[column] = input_dataset[column].astype('category')
         input_dataset[column] = input_dataset[column].cat.codes


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result




# Normalize the data
input_dataset = normalize(input_dataset)

outputPath = sys.argv[2] if len(sys.argv) > 1 else "ProcessedFile"
input_dataset.to_csv(outputPath, index=False)