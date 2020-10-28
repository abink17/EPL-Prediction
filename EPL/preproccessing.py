import pandas as pd

stats_data = pd.read_csv("E:\\Projects\\stats.csv")
print(stats_data.head())

#Counting Null Columns
null_col = stats_data.columns[stats_data.isnull().any()]
stats_data[null_col].isnull().sum()

#Impute values