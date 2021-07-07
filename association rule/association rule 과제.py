# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 20:39:58 2021

@author: tkdan
"""

import pandas as pd

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
#print(transactions)


from apyori import apriori
rules = apriori(transactions, min_support = 0.0046, min_confidence = 0.2, min_lift = 4, min_length = 2)

results = list(rules)
#print(results)
df_results = pd.DataFrame(results)

for item in results:
    pair = item[0]
    items = [x for x in pair]
    #print(items)
    
    print("Rule: " + items[0], end='')
    for i in range(1,len(items)):
        print(" -> " + items[i], end='')
    
    print('\n'+"Support: " + str(item[1]))
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("======================================")
    
    