###### Pyodbc package

# https://medium.com/@DrGabrielA81/python-how-connect-to-and-manage-a-database-68b113a5ca62

import pyodbc
import pandas as pd

user = 'UserID'
password = 'password'
host = 'serverName'
database = 'databaseName'
driver = 'SQL Server'

conn = pyodbc.connect(Driver = driver,
                      Server = host,
                      Database = database,
                      UID = user,
                      Password =password)

sql = 'SELECT * FROM [dbo].[table]'

pd.read_sql(sql, conn)



