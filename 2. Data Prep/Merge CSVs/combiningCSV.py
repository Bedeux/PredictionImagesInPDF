import pandas as pd

df1 = pd.read_csv('DataIABB.csv', encoding='latin-1')
df2 = pd.read_csv('DataIA_BG_FINAL.csv')
df3 = pd.read_csv('targetValues.csv')

output = pd.merge(df1, df2, how='inner', on='DocumentName')
final_input = pd.merge(output, df3, how='inner', on='DocumentName')

output.to_csv("final_data", sep='\t', encoding='utf-8')
