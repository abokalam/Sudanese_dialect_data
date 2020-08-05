import numpy as np
import io
import os
import glob
import pandas as pd
from langdetect import detect
import time
import re
import multiprocessing as mp


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

print("script execution begins")

#mycsvdir = '/home/muhammed/Desktop/MTN/chatbot/clean_data_test'
#csvfiles = glob.glob(os.path.join(mycsvdir, '*.csv'))

#pass as input the path of the folder, where the telegram csv files are stored
csvfiles = os.listdir('/home/muhammed/Desktop/MTN/chatbot/clean_data_test')
#this line of code returns, the names - not pathes - of all subdirectories - telegram csv files - in the folder


#for csvfile in csvfiles:
counter2 = 0
def launch(csvfile):
   
    
    start_time = time.time()
    # the path is the same as the above in os.listdir
    csv_path = '/home/muhammed/Desktop/MTN/chatbot/clean_data_test/'+csvfile
    df = pd.read_csv(csv_path)
    df = df.drop(['0'], axis = 1)
    df.columns = ['Time', 'Text', 'User']
    new_list = [['Time' , 'Text' , 'User']]
    df['User'] = df['User'].fillna('temp_nan')
    df['Text'] = df['Text'].fillna('temp_nan')
    df['Time'] = df['Time'].fillna('temp_nan')
    initialized = True
    counter = 0
    User = ''
    for index, row in df.iterrows():
            text_array = row['Text']
            text_array = deEmojify(text_array)
            text_array = text_array.replace('#','')
            text_array = text_array.split()
            filtered_text_array = []
            #if counter > 490:
            #      break
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
        
    
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2')
    print(csvfile)
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    new_df = pd.DataFrame(new_list,columns=['Time','Text', 'User'])
    new_list.clear()
    print('saving file : ' + csvfile + '   cleaning time: ' + str(time.time() - start_time))
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    
    file_name = csvfile + '_cleaned' + '.csv'
    # print(User)
    
    
    #change the path of hte file depend on where you want to store it
    path = "/home/muhammed/Desktop/MTN/chatbot/clean_data_test_after_mukho_code/"+file_name
    
    print('new file path is '+path)
    print('.....................................................................')
    new_df.to_csv(path)



if __name__ =='__main__':
    print(csvfiles)
    #Note
    #open python script 
    #then import multiprocessing as mp
    #then check the total number of processers you have in your machine by writing the following code
    #first mp.cpu_count()
    #make sure num_processes is less than mp.cpu_count() else you will be running multithreading and this will reduce the speed of multiprocessing
    #counter2 = 0
    num_processes = 10
    processes = []
    #for rank in range(num_processes):
    pool = mp.Pool(num_processes)
    pool.map(launch, csvfiles)





    
   # file_name = csvfile.split('.')
   # file_name = file_name[0]
   # file_name = file_name.split('/')   
   # file_name = file_name[len(file_name) - 1]
   # print('saving file : ' + file_name)

