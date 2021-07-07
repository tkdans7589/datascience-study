# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:14:52 2021

@author: tkdan
"""

from collections import defaultdict, Counter
from typing import List, NamedTuple, Tuple, Dict
from scratch.linear_algebra import Vector, distance
from scratch.machine_learning import split_data
import random
import pandas as pd
from pprint import pprint

def majority_vote(labels: List[str]) -> str:
    """Assumes that labels are ordered from nearest to farthest."""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])

    if num_winners == 1:
        return winner                     # unique winner, so return it
    else:
        return majority_vote(labels[:-1]) # try again without the farthest

class LabeledPoint(NamedTuple):
    point: Vector
    label: str

def knn_classify(k: int,
                 labeled_points: List[LabeledPoint],
                 new_point: Vector) -> str:

    # Order the labeled points from nearest to farthest.
    by_distance = sorted(labeled_points,
                         key=lambda lp: distance(lp.point[1:], new_point[1:]))
    
    # Find the labels for the k closest
    k_nearest_labels = [lp.label for lp in by_distance[:k]]
    
    # and let them vote.
    return majority_vote(k_nearest_labels)

df=pd.read_csv('bmd.csv')
df2= df[['id','age', 'weight_kg', 'height_cm', 'bmd', 'sex']]
pd.set_option('mode.chained_assignment',  None)
for i in range(len(df2)):
    if df2['sex'].iloc[i] == 'F':
        df2['sex'].iloc[i] = 1
    else :
        df2['sex'].iloc[i] = 2

measurements=[]
label=[]

for i in range(len(df2)):
    measurements.append(list(df2.iloc[i]))
    label.append(df['fracture'].iloc[i])

bmd_data=[LabeledPoint(i,j) for i,j in zip(measurements, label)]

points_by_species: Dict[str, List[Vector]] = defaultdict(list)
for bmd in bmd_data:
    points_by_species[bmd.label].append(bmd.point[1:])

random.seed(30)
bmd_train, bmd_test = split_data(bmd_data, 0.75)
confusion_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
#print("bmd_train\n",bmd_train,'\n')
#print("bmd_test\n", bmd_test, '\n')

max_dict=pd.DataFrame()
max_accuracy=0
max_accuracy_K=0

for K in [1,3,5,7]:
    num_correct = 0
    tp=0
    fp=0
    fn=0
    
    predicted_data=pd.DataFrame(columns=['id', 'predicted', 'actual'])
    
    for bmd in bmd_test:
        predicted = knn_classify(K, bmd_train, bmd.point)
        actual = bmd.label
        predicted_data=predicted_data.append(pd.Series([bmd.point[0],predicted, actual], index=predicted_data.columns), ignore_index=True)
    
        if predicted == actual:
            num_correct += 1
    
        confusion_matrix[(predicted, actual)] += 1
        tp=confusion_matrix[('fracture', 'fracture')]
        fp=confusion_matrix[('fracture', 'no fracture')]
        fn=confusion_matrix[('no fracture', 'fracture')]
    
    pct_correct = num_correct / len(bmd_test)
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    
    if pct_correct > max_accuracy:
        max_accuracy = pct_correct
        max_dict = predicted_data
        max_accuracy_K = K
    
    print("K =", K, ", accuracy :",pct_correct, ", precision :", precision, ", recall :", recall)

print("\nK = ",max_accuracy_K,"일 때 최적\n")
print("accuracy =",max_accuracy,'\n')
pprint(max_dict.iloc[-10:])

from matplotlib import pyplot as plt
metrics = ['age', 'weight_kg', 'height_cm', 'bmd', 'sex']
pairs = [(i, j) for i in range(5) for j in range(5) if i < j]
marks = ['+', 'x']

fig, ax = plt.subplots(2, 5)

for row in range(2):
    for col in range(5):
        i, j = pairs[5 * row + col]
        ax[row][col].set_title(f"{metrics[i]} vs {metrics[j]}", fontsize=8)
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks([])

        for mark, (species, points) in zip(marks, points_by_species.items()):
            xs = [point[i] for point in points]
            ys = [point[j] for point in points]
            ax[row][col].scatter(xs, ys, marker=mark, label=species)

ax[-1][-1].legend(loc='lower right', prop={'size': 6})
plt.show()
