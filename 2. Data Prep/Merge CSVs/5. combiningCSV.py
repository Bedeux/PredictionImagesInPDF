import pandas as pd

"""Create one CSV with all variables we create independantly"""

df1 = pd.read_csv('colors_variables.csv', encoding='latin-1')
df2 = pd.read_csv('characters_variables.csv')
df3 = pd.read_csv('target_values.csv')

output = pd.merge(df1, df2, how='inner', on='DocumentName')
final_input = pd.merge(output, df3, how='inner', on='DocumentName')

output.to_csv("final_data", sep='\t', encoding='utf-8')
