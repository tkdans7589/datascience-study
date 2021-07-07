# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 20:40:59 2021

@author: tkdan
"""
#1번
import seaborn as sns

titanic = sns.load_dataset("titanic") # returns DataFrame
print('1번\n', titanic)

#2번
columns=['sex', 'age', 'class', 'alive']
print('2번\n')
for t in columns:
    print(titanic[t].value_counts(), '\n')

#3번

#3-1번
df = titanic[['age', 'embark_town', 'sex', 'class']].set_index(['sex', 'class'])
print('3-1번\n',df)

#3-2번
df = df.sort_values(['sex', 'class'])
print('3-2번\n', df)