# put your (.html) files in Telegram_raw_data file
# the output will be at the same directory as this file
import codecs
import pandas as pd
import glob
import emojis

list_of_files = glob.glob('Telegram_raw_data/*.html')           # create the list of file

count = 0
is_text = False
is_header = False
header = 'NO_HEADER'

for file_name in list_of_files:
  df = pd.DataFrame(columns=['Datetime', 'Text', 'username'])
  print(file_name)
  f=codecs.open(file_name,'r',encoding="utf8")
  Lines = f.readlines()
  for line in Lines:
     if "</div>" in line:
         is_text = False
     if is_header:
         header = line
         header = header.replace('\n', '')
         header = header.replace(' ', '_')
         is_header = False
         
     if is_text:
         # print(line)
         line = line.replace('\n' , '<br>')
         line = line.replace('..' , '<br>')
         line_emojis = emojis.get(line)
         for an_emoji in line_emojis:
             line.replace(an_emoji ,'<br>' )
         splitted_line = line.split('<br>')
         for a_line in splitted_line:
            if a_line == '':
               continue
            count += 1
            output_list = ['NO_DATE' , a_line , header]
            a_series = pd.Series(output_list, index = df.columns)
            df = df.append(a_series, ignore_index=True)
         # is_text = False

   
     if "<div class=\"text\">" in line:
         is_text = True
     if "<div class=\"text bold\">" in line:
         is_header = True
  print(count)
  df.to_csv(header+'.csv', encoding='utf-8' , mode='a', header=False)

print(count)
