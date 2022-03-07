###
## Example Market Basket Analysis
## 2019 5 augustus
## 
## https://pbpython.com/market-basket-analysis.html

# change working directory
#import os
#os.getcwd()
#os.getcwd()

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

######
# get some packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# options to show more columns
pd.set_option('display.max_columns', 100)

#####################
input_file = ''

df = pd.read_excel(input_file)
df.head()
df.dtypes

# Clean up spaces in description and remove any rows that don't have a valid invoice
df['Description'] = df['Description'].str.strip()
df['Description'].head()

df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]

df.head()


### make basket
basket = (df[df['Country'] =="France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

basket.head()

# Show a subset of columns
basket.iloc[:,[0,1,2,3,4,5,6, 7]].head()

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)

# No need to track postage
basket_sets.drop('POSTAGE', inplace=True, axis=1)

basket_sets.head()

# Build up the frequent items
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)

frequent_itemsets.head()

# Create the rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules

#### check out rules that score good
rules[ (rules['lift'] >= 6) &
       (rules['confidence'] >= 0.8) ]

