# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:41:58 2018

@author: Gijs
"""

import pandas as pd
import numpy as np

# create the dataframe with a dictionary
dict1 = dict({'col1': [1, 2, 3, 4, 5, 4, 3, 2, 1, 999, 1, 999, 3, 4, -999], 'col2': [999, 1, 1, 2, 2, 2, 3, 3, 3, 999, 4, 4, 5, 5, 5] })

df1 = pd.DataFrame(dict1)

print(df1)

# prints a new series
print(df1['col1'].replace(999, np.nan))

# prints the whole dataframe
print(df1.replace(999, np.nan))

# recode multiple values
print(df1.replace([-999, 999], np.nan))

# different replacements for different values
print(df1.replace([-999, 999], [0, np.nan]))

# reverse values (1 becomes 5, 2 becomes 4, 4 becomes 2, 5 becomes 1)
print(df1.replace([-999, 999, 1, 2, 3, 4, 5], [np.nan, np.nan, 5, 4, 3, 2, 1]))

# add new column instead of replacing
df1['col3'] = df1['col1'].replace([-999, 999, 1, 2, 3, 4, 5], [np.nan, np.nan, 5, 4, 3, 2, 1])
print(df1)

# reset
df1 = pd.DataFrame(dict1)

# add multiple columns instead of only one
df1[['col3', 'col4']] = df1[['col1', 'col2']].replace([-999, 999, 1, 2, 3, 4, 5], [np.nan, np.nan, 5, 4, 3, 2, 1])
print(df1)

# reset 
df1 = pd.DataFrame(dict1)

# alternatively, add multiple columns instead of only one
df1[['col3', 'col4']] = df1.replace([-999, 999, 1, 2, 3, 4, 5], [np.nan, np.nan, 5, 4, 3, 2, 1])
print(df1)

# reset 
df1 = pd.DataFrame(dict1)

# use a dictionary instead of lists
dict_recode = dict({-999: np.nan, 999: np.nan, 1:5, 2:4, 3:3, 4:2, 5:1})
df1[['col3', 'col4']] = df1.replace(dict_recode)
print(df1)

# inplace replacing instead of having to specify new column
print(df1.col1)
df1.col1.replace(dict_recode, inplace=True)
print(df1.col1)
# reset 
df1 = pd.DataFrame(dict1)

# aleternatively
print(df1.col1)
df1.col1 = df1.col1.replace(dict_recode)
print(df1.col1)
# reset 
df1 = pd.DataFrame(dict1)