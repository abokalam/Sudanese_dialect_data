# put your raw file (.txt) in the same directory as this file and change filename at line 7 to the name of the raw file
#change the name inside df.to_csv() to the name you want for your output file

import re
import pandas as pd
import emoji

file1 = open('input_file_name.txt', 'r', encoding="utf8") 
Lines = file1.readlines() 
  
count = 0
print(len(Lines))

df = pd.DataFrame(columns=['Datetime', 'Text', 'username'])

# Strips the newline character 
for line in Lines: 
    count = count + 1
    
    if line[0] in emoji.UNICODE_EMOJI:
       continue
    
    if line[0].isdigit():
       # print("starts with a digit: " + line)
       new_line = re.split(': | - ',line)
       if len(new_line)<3:
          new_line = ['']
          new_line.append(line)
          new_line.append('')
       if len(new_line)>3:
            continue
       # print(new_line)
    else:
       new_line = ['']
       new_line.append('')
       new_line.append(line)
    
    if new_line[2] == '' or new_line[2] == '\n'or new_line[2] == '<Media omitted>\n':
       continue

       
    # print(new_line)
    output_line = [new_line[0] , new_line[2] , new_line[1]]
    a_series = pd.Series(new_line, index = df.columns)
    
    df = df.append(a_series, ignore_index=True)

print(count)
df.to_csv('output_file_name.csv', encoding='utf-8')