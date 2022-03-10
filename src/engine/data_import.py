
import numpy as np
import pandas as pd

path_to_file = r"C:\Users\Anett\Documents\Msc\PPKE\3. semester\Brain Therapy Tech. (Dani)\Retina\Data from first visit\CB07_970322_220216105506_Small.csv"
with open (path_to_file, "r") as f:
    content = f.read()
    elements = content.split(',')
    myarray = np.array(elements)
    # myarray_2d = myarray.reshape ((-1,36))
    """
    itt 2dimenziós tömbök akartam létrehozni az 1-ből de nem működött
    """
    print(myarray[:100])








# encoding ='latin1'
# Df = pd.read_csv(path_to_file, encoding="latin1", low_memory=False)
# data = pd.read_table(path_to_file,skip_blank_lines=True, na_filter=True)
# df=pd.read_csv(path_to_file).dropna()
"""
üres sorok kidobása
"""
