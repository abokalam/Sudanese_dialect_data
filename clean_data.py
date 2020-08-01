# to run this script put your csv files in a file called 'input_file' and create a file called 'clean_data_output_file' wherer your outputs will be placed
# your input file should have the header ['Time' , 'Text' , 'User']
import numpy as np
import io
import os
import glob
import pandas as pd
from langdetect import detect
import time
import re


def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F6FF"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF"
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

print("script execution begins (new code)")

mycsvdir = 'input_file'
csvfiles = glob.glob(os.path.join(mycsvdir, '*.csv'))

counter2 = 0
for csvfile in csvfiles:
   start_time = time.time()
   print(csvfile)
   df = pd.read_csv(csvfile)
   new_list = [['Time' , 'Text' , 'User']]
   initialized = True
   counter = 0

   for index, row in df.iterrows():
         text_array = row['Text']
         text_array = deEmojify(text_array)
         text_array = text_array.replace('#','')
         text_array = text_array.split()
         filtered_text_array = []
         # if counter > 490:
            # break
         for word in text_array:
            try:
              lang = detect(word)
             #  print(lang)
              if lang=='ar' or lang=='fa' or lang=='ur':
               #  print(word + '   :  added')
                filtered_text_array.append(word)
               #  else:
               #  print(word + '   :  deleted')
            except Exception:
             #  print("#"+word + '   :  deleted')
              pass
         new_text_array = " ".join(filtered_text_array)
         # print("////////"+new_text_array)
         counter += 1
         counter2 += 1
         if counter % 10000 == 0:
              print(counter)
         # print(counter)
         if row['User'] != 'User': 
              User = row['User']
         # if (initialized == True):
         #   new_list = [[row['Time'] , new_text_array , row['User']]]
         #   initialized == False
         # else:
         if new_text_array != '':
            new_list.append([row['Time'] , new_text_array , row['User']])
         # print(len(new_list))
      
   new_df = pd.DataFrame(new_list,columns=['Time','Text', 'User'])
   new_list.clear()
   print('saving file : ' + User + '   number of lines :' + str(counter) + '  cleaning time: ' + str(time.time() - start_time))
   file_name = User + '_cleaned' + '.csv'
   # print(User)
   path = "clean_data_output_file/" + file_name
   new_df.to_csv(path)
print('number of lines : ' +  str(counter2))
   # file_name = csvfile.split('.')
   # file_name = file_name[0]
   # file_name = file_name.split('/')   
   # file_name = file_name[len(file_name) - 1]
   # print('saving file : ' + file_name)

