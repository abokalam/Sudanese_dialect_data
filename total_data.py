import os
import pandas as pd

folders = os.listdir('/home/muhammed/Desktop/MTN/chatbot/clean_data_test_after_mukho_code/')

total = 0
print(folders)
for folder in folders:
    
    path = '/home/muhammed/Desktop/MTN/chatbot/clean_data_test_after_mukho_code/'+folder
    try:
        df = pd.read_csv(path)
        
        total += len(df)
        print('------------------------')
        print('file name ')
        print(folder)
        print('#########################')
        print(len(df))
        print('**********************')
    except:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
        print(folder)
print('total len equals ')
print(total)
