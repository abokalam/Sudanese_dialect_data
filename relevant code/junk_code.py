import numpy as np
import io
import os
import glob
import pandas as pd
from langdetect import detect
import time
import re
import random





start_time = time.time()
lineList = [line.rstrip('\n') for line in open('Sudanese_txt_cleaned_shuffled_all.txt', encoding="utf-8")]

new_list = lineList[1:4000]


with open('slice.txt', 'a',encoding="utf-8") as filehandle:
   filehandle.writelines("%s\n" % line for line in new_list)






   
def file_lengthy(fname):
        with open(fname, encoding="utf-8") as f:
                for i, l in enumerate(f):
                        pass
        return i + 1
print("number of lines in file : ", file_lengthy("slice.txt"))
