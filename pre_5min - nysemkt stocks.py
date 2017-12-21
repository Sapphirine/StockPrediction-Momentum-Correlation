import os
import numpy as np
import shutil




dir = (os.path.dirname(os.path.realpath(__file__)))+"\\5_us_txt\\data\\5 min\\us\\"


folders = sorted(os.listdir(dir))

folders_path = [os.path.join(dir,oneFolder) for oneFolder in folders]
stocks = []


for eachFolder in folders_path:
    if eachFolder.endswith('nysemkt stocks'):
        for oneFile in os.listdir(eachFolder):
            path = os.path.join(eachFolder,oneFile)
            if os.path.getsize(path) >= 10240:
                stocks.append(path)
import pickle
with open('stocks_path.txt','wb+') as savefile:
    pickle.dump(stocks,savefile)
    
# Test that the path file is loadable
with open('stocks_path.txt','rb') as savefile:
    stocks_load = pickle.load(savefile)
