import pandas as pd

data=pd.read_csv('train.txt', sep=' ', header=None, names=['token', 'POS', 'chunking', 'chunking2'])
data=data[['token','POS']]
print(data['POS'].unique())

'''
features:
- end in s
- end in ing
- end in ed
- word length
- end in 's
- a, an, the
- end in ly
- in dictionary
- capitalized
- alpha or num

'''
