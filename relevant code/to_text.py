import numpy as np
import io
import os
import glob
import pandas as pd
from langdetect import detect
import time
import re
import random



print("script execution begins (new code)")

mycsvdir = 'twitter_files_cleaned'
csvfiles = glob.glob(os.path.join(mycsvdir, '*.csv'))

output_list = []
adder = 0

for csvfile in csvfiles:
   start_time = time.time()
   print(csvfile)
   df = pd.read_csv(csvfile)
   col_one_list = df['Text'].tolist()
   adder += len(col_one_list)
   # output_list.append(col_one_list)
   with open('Sudanese_txt_cleaned_twitter.txt', 'a', encoding="utf-8") as filehandle:
      filehandle.writelines("%s\n" % line for line in col_one_list)

print('number of lines in list : '+str(adder))

def file_lengthy(fname):
        with open(fname, encoding="utf-8") as f:
                for i, l in enumerate(f):
                        pass
        return i + 1
print("number of lines in file : ", file_lengthy("Sudanese_txt_cleaned_twitter.txt"))



   