import pandas as pd
data = pd.read_csv('births.csv')
print(data.head())

data["decade"] = 10 * (data["year"]//10)
print(data.head())

data.pivot_table('births', index='decade', columns='gender', aggfunc='sum').plot(title='Births by Decade and Gender')