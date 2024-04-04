import pandas as pd

data=pd.read_csv("gps-ops.csv")
print(data.loc[30])
print(data["EPOCH"])