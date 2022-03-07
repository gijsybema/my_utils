# import packages for fuzzy string matching
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Explanation of fuzzywuzz package  
# https://www.datacamp.com/community/tutorials/fuzzy-string-python 

# tests

## Part 1. Find how similar two strings are
test_1 = 'NIEUWERKERK AD IJSSEL'
test_2 = 'NIEUWERKERK AAN DEN IJSSEL'

# Different ways
# 1. Levenshtein distance
Ratio = fuzz.ratio(test_1.lower(), test_2.lower())

# 2. optimal partial logic
# if the short string has length k and the longer string has the length m, then the algorithm seeks the score of the best matching length-k substring.
Partial_Ratio = fuzz.partial_ratio(test_1.lower(), test_2.lower())

# 3. token sort ratio
# They tokenize the strings and preprocess them by turning them to lower case and getting rid of punctuation. 
# In the case of fuzz.token_sort_ratio(), the string tokens get sorted alphabetically and then joined together
Token_Sort_Ratio = fuzz.token_sort_ratio(test_1, test_2)

# 4. Token set ratio
Token_Set_Ratio = fuzz.token_set_ratio(test_1, test_2)

# show ratios
print(Ratio)
print(Partial_Ratio)
print(Token_Sort_Ratio)
print(Token_Sort_Ratio)

## Part 2: check which string is closest from a list of strings 
## Tests
str2Match = 'NIEUWERKERK AD IJSSEL'
strOptions = ['NIEUWERKERK AAN DEN IJSSEL', 'NIEUWERKERK', 'ELSE']
Ratios = process.extract(str2Match, strOptions, scorer = fuzz.token_sort_ratio)
print(Ratios)

# Select the string with the highest matching percentage
highest = process.extractOne(str2Match, strOptions, scorer = fuzz.token_sort_ratio)
print(highest)