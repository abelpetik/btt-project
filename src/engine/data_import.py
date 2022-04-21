import numpy as np
import pandas as pd
import  tkinter as tk
import pickle

def import_data():
    """
    Kiválaszhatod melyik file-t nyitod meg, majd az adatot list_of_dicts néven elérheted,
    amelyek tartalmazzák 2 oszloponként dictionary-ben a patient/measurement alap adatokat,
    ,valamint a számértékeket 2D numpy array formában, amik a Reported Waveform[ms,uV], Raw Waveform[ms,uV], Pupil Waveform[ms,mm] dictionary értékei.
    A program mindig kiprinteli az adott file-ban található oszlopok alapján létrehozott dictionary-k számát is,
    valamint emlékeztetőnek milyen sorrendben vannak a mértékegységek.

    Returns
    -------

    """

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
    drop_cols = (data[data >= limit].index)
    new_df = df.drop (columns = drop_cols)

    df_shape = new_df.shape
    # print(df_shape)
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

    unit_loc_inc = list(np.asarray(unit_location) + 1)

    # print(unit_location)
    # print(unit_loc_inc)


    #Mértékegységekhez tartozó adatpárok arraybe, majd dictionarybe tevése, végül hozzácsatolás az oszlop Patient adatokat tartalmazó dictionaryhoz
    #Reported Waveform dataset [ms,uV]

    a = 0
    j = 1
    for i in range(2, df_shape[1], 2):

        list1 = new_df[f'{i}'].tolist()
        del list1[:unit_loc_inc[0]]
        del list1[unit_loc_inc[1]:]

        j +=2
        list2 = new_df[f'{j}'].tolist()
        del list2[:unit_loc_inc[0]]
        del list2[unit_loc_inc[1]:]

        key = new_df.at[unit_location[0],"0"]
        unit_values_2d = np.array((list1,list2))
        new_dict = {}
        new_dict[key] = unit_values_2d

        list_of_dicts[a].update(new_dict)
        a += 1

    print("Reported Waveform measurement units: [ms,uV]")


    #Raw Waveform dataset [ms,uV]

    a = 0
    j = 1
    for i in range(2, df_shape[1], 2):
        list1 = new_df[f'{i}'].tolist()
        del list1[:unit_loc_inc[1]]
        del list1[unit_loc_inc[2]:]

        j +=2
        list2 = new_df[f'{j}'].tolist()
        del list2[:unit_loc_inc[1]]
        del list2[unit_loc_inc[2]:]

        key = new_df.at[unit_location[1],"0"]
        unit_values_2d = np.array((list1,list2))
        new_dict = {}
        new_dict[key] = unit_values_2d
        list_of_dicts[a].update(new_dict)
        a += 1

    print("Raw Waveform measurement units: [ms,uV]")


    #Pupil Waveform dataset [ms,mm]
    a = 0
    j = 1
    for i in range(2, df_shape[1], 2):
        list1 = new_df[f'{i}'].tolist()
        del list1[:unit_loc_inc[2]]

        j +=2
        list2 = new_df[f'{j}'].tolist()
        del list2[:unit_loc_inc[2]]

        key = new_df.at[unit_location[2],"0"]
        unit_values_2d = np.array((list1,list2))
        new_dict = {}
        new_dict[key] = unit_values_2d
        list_of_dicts[a].update(new_dict)
        a += 1

    print("Pupil Waveform measurement units: [ms,mm]")
    print(f"You can access the data using the following labels: {list_of_dicts[0].keys()}")
    print(f"Numbers of dictionaries: {len(list_of_dicts)}")

    return list_of_dicts

def save(final_data):

    # nem tudom hogy ez így működik-e

    with open('filename.pickle', 'wb') as handle:
        pickle.dump(final_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return final_data

def load(file_name):

    with open(file_name, 'rb') as handle:
        b = pickle.load(handle)

    return b

# Lekérdezés tesztelése
# print(list_of_dicts[3]['Reported Waveform'])
# print(list_of_dicts[3]['Reported Waveform'][0])

# print(list_of_dicts[1]['Raw Waveform'])
# print(list_of_dicts[1]['Raw Waveform'][1])

# print(list_of_dicts[0]['Pupil Waveform'])
# print(list_of_dicts[0]['Pupil Waveform'][0])
# print(list_of_dicts[0]['Pupil Waveform'][0][0])
