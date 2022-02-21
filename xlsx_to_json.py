import pandas as pd
import os
import json

pd.set_option("display.max_columns", 20)

# Read excel document
project_folder = 'C:\\Users\\A0782050\\PycharmProjects\\20220221 - Excel to json script python'
input_folder = 'data'
output_folder = 'output'
input_file = '20220216 Nexperia location.xlsx'
output_file = '20220216 Nexperia location.json'

# make strings of input and output file location
input_file = os.path.join(project_folder, input_folder, input_file)
output_file = os.path.join(project_folder, output_folder, output_file)

# load excel file to dataframe
excel_data_df = pd.read_excel(input_file)#, sheet_name='sheet1')
print(excel_data_df.info())

# Convert excel to string
# (define orientation of document in this case from up to down)
thisisjson = excel_data_df.to_json(orient='records', date_format='iso', double_precision=15)

# Print out the result
#print('Excel Sheet to JSON:\n', thisisjson)

# Make the string into a list to be able to input in to a JSON-file
thisisjson_dict = json.loads(thisisjson)
#print(thisisjson_dict)

# Define file to write to and 'w' for write option -> json.dump()
# defining the list to write from and file to write to
data_json = json.dumps(thisisjson_dict, indent=4)

with open(output_file, 'w') as f:
    f.write(data_json)

# check if json file contains same information as excel file
json_check_df = pd.read_json(output_file)

print('Print shapes and assert if equal')
print(excel_data_df.shape)
print(json_check_df.shape)
assert excel_data_df.shape == json_check_df.shape, 'Dataframe shapes are not equal'

# datetime column is in another format
df1 = excel_data_df.loc[:, excel_data_df.columns != 'TRANSACTION_DATE_TIME']
df2 = json_check_df.loc[:, json_check_df.columns != 'TRANSACTION_DATE_TIME']
#df1 = excel_data_df
#df2 = json_check_df

print('Print first rows and assert if dataframes are equal')
print(df1.head())
print(df2.head())
for col in df1.columns:
    print(col)
    assert df1[col].equals(df2[col]), 'Column values are not equal'
#assert df1.equals(df2), 'Dataframes are not equal'

print('Completed')

