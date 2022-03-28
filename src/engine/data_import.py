import numpy as np
import pandas as pd
import  tkinter as tk

from tkinter import filedialog
root = tk.Tk()
root.withdraw()
path_to_file = filedialog.askopenfilename()
print(path_to_file)


df = pd.read_csv (path_to_file, sep= ",", encoding = 'unicode_escape', header=None, low_memory=False)
data = df.isna().sum() # Detect missing values. Count the NaN values in columns.
empty = data.values.tolist()
max_value = max(empty)
limit = max_value-24
ind = (data[data >= limit].index)
drop_cols = df.drop ( columns = ind )

df_shape = drop_cols.shape
print (df_shape)

col_numbs = list(range(0, df_shape[1]))
col_numbs_str = map(str, col_numbs)
col_names = list(col_numbs_str)
drop_cols.columns = col_names

path_to_file2 = path_to_file + str("new") + str(".csv")
print (path_to_file2)
drop_cols.to_csv(path_to_file2, sep=",", encoding='Latin-1', index=False, header=False)
print("Saved new data!")

#Dictionary létrehozás listákból
Labels = drop_cols['0'].tolist()
del Labels[26:]
print(Labels)

my_list = list()

for i in range(2,df_shape[1]):
    a=2
    col_3_list = drop_cols[f'{a}'].tolist()
    del col_3_list[26:]
    print(col_3_list)

    base_dict = dict(zip(Labels, col_3_list))
    print(base_dict)
    base_dict.items()


dict_lists = list()
b = 4 # 4 mivel a 0,1,2 -> 2. oszlop adataival létrehoztam a dictionaryt

for i in range(28):
    for j in range(df_shape[1]):
        cell = drop_cols.iloc[i,b]
        if pd.isna(cell):
            b += 1
        else:
            dict_lists.append(cell)
            b += 1

        #dict[key].extend(list of values) ez lenne a cél hogy a soronként létrehozott listákat beletenni az első sor dictionary key-be,

print(dict_lists)