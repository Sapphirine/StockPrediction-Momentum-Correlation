import os
import numpy as np
import shutil




dir = (os.path.dirname(os.path.realpath(__file__)))+"\\5_us_txt\\data\\5 min\\us\\"

if os.path.isdir(os.path.join(dir, 'nasdaq stocks\\1')):
    for eachFile in os.listdir(os.path.join(dir, 'nasdaq stocks\\1')):
        shutil.move(dir+'nasdaq stocks\\1\\'+eachFile,dir+'nasdaq stocks\\')
    os.rmdir(os.path.join(dir, 'nasdaq stocks\\1\\'))
    for eachFile in os.listdir(os.path.join(dir, 'nasdaq stocks\\2')):
        shutil.move(dir+'nasdaq stocks\\2\\'+eachFile,dir+'nasdaq stocks\\')
    os.rmdir(os.path.join(dir, 'nasdaq stocks\\2\\'))
if os.path.isdir(os.path.join(dir, 'nyse stocks\\1')):
    for eachFile in os.listdir(os.path.join(dir, 'nyse stocks\\1')):
        shutil.move(dir+'nyse stocks\\1\\'+eachFile,dir+'nyse stocks\\')
    os.rmdir(os.path.join(dir, 'nyse stocks\\1\\'))
    for eachFile in os.listdir(os.path.join(dir, 'nyse stocks\\2')):
        shutil.move(dir+'nyse stocks\\2\\'+eachFile,dir+'nyse stocks\\')
    os.rmdir(os.path.join(dir, 'nyse stocks\\2\\'))




folders = sorted(os.listdir(dir))

folders_path = [os.path.join(dir,oneFolder) for oneFolder in folders]
stocks = []


for eachFolder in folders_path:
    for oneFile in os.listdir(eachFolder):
        path = os.path.join(eachFolder,oneFile)
        if os.path.getsize(path) >= 10240:
            stocks.append(path)
import pickle
with open('stocks_path.txt','wb+') as savefile:
    pickle.dump(stocks,savefile)

with open('stocks_path.txt','rb') as savefile:
    stocks_load = pickle.load(savefile)
