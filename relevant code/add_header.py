import numpy as np
import io
import os
import glob
import pandas as pd
from langdetect import detect
import time
import re




print("script execution begins")

mycsvdir = 'input_file'
csvfiles = glob.glob(os.path.join(mycsvdir, '*.csv'))
counter = 1
counter2 = 0
for csvfile in csvfiles:
   start_time = time.time()
   print(csvfile)
   df = pd.read_csv(csvfile , encoding = 'utf-8', header=None)
   df.columns  = ['index','Time' , 'Text' , 'User']
   path = "output_file/"+ str(counter)+'.csv'
   df.to_csv(path)
   counter += 1
   
   
