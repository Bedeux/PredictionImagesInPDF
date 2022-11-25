import pandas as pd

df1 = pd.read_csv('DataIABB.csv', encoding='latin-1')
df2 = pd.read_csv('DataIABG.csv', encoding='latin-1')
df3 = pd.read_csv('targetValues.csv', encoding='latin-1')

output = pd.merge(df1, df2, how='inner', on='DocumentName')
output = pd.merge(output, df3, how='inner', on='DocumentName')

output.to_csv("final_data", sep=',', encoding='utf-8')
