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

mytxtdir = 'Sudanese_txt_cleaned'
txtfiles = glob.glob(os.path.join(mytxtdir, '*.txt'))

output_list = []
adder = 0

for txtfile in txtfiles:
   start_time = time.time()
   print(txtfile)
   lineList = [line.rstrip('\n') for line in open(txtfile, encoding="utf-8")]
   adder += len(lineList)
   with open('Sudanese_txt_cleaned_all.txt', 'a',encoding="utf-8") as filehandle:
      filehandle.writelines("%s\n" % line for line in lineList)
   


print('number of lines in list : '+str(adder))


   
def file_lengthy(fname):
        with open(fname, encoding="utf-8") as f:
                for i, l in enumerate(f):
                        pass
        return i + 1
print("number of lines in file : ", file_lengthy("Sudanese_txt_cleaned_all.txt"))
