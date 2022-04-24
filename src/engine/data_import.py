import numpy as np
import pandas as pd
import  tkinter as tk
from tkinter import filedialog

def importer():
    root = tk.Tk()
    root.withdraw()
    path_to_file = filedialog.askopenfilename()
    print(path_to_file)

    df = pd.read_csv (path_to_file, sep= ",", encoding = 'unicode_escape', header=None, low_memory=False)
    data = df.isna().sum() # Detect missing values. Count the NaN values in columns.
    empty = data.values.tolist()
    max_value = max(empty)
    limit = max_value-24
    drop_cols = (data[data >= limit].index)
    new_df = df.drop (columns = drop_cols)

    df_shape = new_df.shape
    print(df_shape)

    col_numbs = list(range(0, df_shape[1]))
    col_numbs_str = map(str, col_numbs)
    col_names = list(col_numbs_str)
    new_df.columns = col_names #Oszlopok újranevezése

    path_to_file2 = path_to_file + str("new") + str(".csv")
    new_df.to_csv(path_to_file2, sep=",", encoding='Latin-1', index=False, header=False)
    print("Saved new data!")


    #Patient/mérési adatok dictionarybe tevése
    labels = new_df['0'].tolist()
    del labels[26:]
    print(labels)

    list_of_dicts = list()

    for i in range(2,df_shape[1],2):
        col_list = new_df[f'{i}'].tolist()
        del col_list[26:]
        data_pair_dict = dict(zip(labels, col_list))
        list_of_dicts.append(data_pair_dict)


    #Mértékegységet tartalmazó sorok megkeresése ("ms")
    unit_location = []

    for loc, unit in enumerate(new_df["2"]):
        if unit == "ms":
            unit_location.append(loc)
    print(unit_location)

    #Mértékegységekhez tartozó adatpárok arraybe, majd dictionarybe tevése, végül hozzácsatolás a Patient adatokat tartalmazó dictionary listához
    #Reported Waveform data
    a = 0
    j = 1
    for i in range(2, df_shape[1], 2):

        list1 = new_df[f'{i}'].tolist()
        del list1[:unit_location[0]]
        del list1[unit_location[1]:]

        j +=2
        list2 = new_df[f'{j}'].tolist()
        del list2[:unit_location[0]]
        del list2[unit_location[1]:]

        key = new_df.at[unit_location[0],"0"]
        unit_values_2d = np.array((list1,list2))
        new_dict = {}
        new_dict[key] = unit_values_2d

        list_of_dicts[a].update(new_dict)
        a += 1

    #Lekérdezés tesztelése
    print(list_of_dicts[3].keys())
    # print(list_of_dicts[3]['Reported Waveform'])
    # print(list_of_dicts[3]['Reported Waveform'][0])

    #Raw Waveform dataset

    a = 0
    j = 1
    for i in range(2, df_shape[1], 2):
        list1 = new_df[f'{i}'].tolist()
        del list1[:unit_location[1]]
        del list1[unit_location[2]:]

        j +=2
        list2 = new_df[f'{j}'].tolist()
        del list2[:unit_location[1]]
        del list2[unit_location[2]:]

        key = new_df.at[unit_location[1],"0"]
        unit_values_2d = np.array((list1,list2))
        new_dict = {}
        new_dict[key] = unit_values_2d
        list_of_dicts[a].update(new_dict)
        a += 1

    # Lekérdezés tesztelése
    print(list_of_dicts[3].keys())
    # print(list_of_dicts[3]['Raw Waveform'])

    #Pupil Waveform dataset
    a = 0
    j = 1
    for i in range(2, df_shape[1], 2):
        list1 = new_df[f'{i}'].tolist()
        del list1[:unit_location[2]]

        j +=2
        list2 = new_df[f'{j}'].tolist()
        del list2[:unit_location[2]]

        key = new_df.at[unit_location[2],"0"]
        unit_values_2d = np.array((list1,list2))
        new_dict = {}
        new_dict[key] = unit_values_2d
        list_of_dicts[a].update(new_dict)
        a += 1

    #Lekérdezés tesztelése
    #print(list_of_dicts[3].keys())
    # print(list_of_dicts[3]['Pupil Waveform'])

    #El van mentve az összes adat array+dictionary-be, és a Patient/measurement data lista ki lett egészítve
    # a Reported Waveform[ms,uV], Raw Waveform[ms,uV], Pupil Waveform[ms,mm] dictionaryval,
    # "[]" jelzésnek beleírtam milyen mértékegységek vannak az adott dictionaryben
    return list_of_dicts
