import numpy as np
import pandas as pd


path_to_file = r"C:\Users\Anett\Documents\Msc\PPKE\3. semester\Brain Therapy Tech\Retina\Data from first visit\CB07_970322_220216105506_Small.csv"
df = pd.read_csv (path_to_file, sep= ";", encoding = 'unicode_escape',  header= None, low_memory=False)
data = df.isna().sum() # Detect missing values. Count the NaN values in columns
empty = data.values.tolist()
max_value = max(empty)
limit = max_value-20
ind = (data[data >= limit].index)
drop_cols = df.drop ( columns = ind )
drop_cols.to_csv(r"C:\Users\Anett\Documents\Msc\PPKE\3. semester\Brain Therapy Tech\Retina\Data from first visit\CB07_970322_220216105506_Small_new.csv", \
                 sep=",", encoding='Latin-1', index=False, header=False)
print("Saved new data!")
