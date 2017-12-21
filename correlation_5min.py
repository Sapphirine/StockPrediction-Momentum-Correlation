# -*- coding: utf-8 -*-
"""
Created on Sat Dec 09 15:10:11 2017

@author: zuodi
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas
import os
import csv
from datetime import datetime as dt
#import pandas as pd
from sklearn import tree
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.kernel_ridge import KernelRidge
from sklearn import svm
from sklearn import preprocessing
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
os.chdir(os.path.dirname(os.path.realpath(__file__)))
class InsufficientData(Exception):
    '''
    When there is no sufficient data to proceed.
    '''
    def __init__(self, value):
        self.parameter = value
    def __str__(self):
        return repr(self.parameter)

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
# Reference: https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array 

    return np.isnan(y), lambda z: z.nonzero()[0]

dataset = {}
with open('stocks_path.txt','rb') as savefile:
    stocks = pickle.load(savefile)

# Load each txt file
i = 0   
print 'Loading dataset' 
for stock in stocks:
    oneStockData= []
    
    with open(stock,'rb') as txtfile:
        oneStockFile = csv.reader(txtfile)
        
        for oneRow in oneStockFile:
            if oneRow[0][0] == '2':         # If it starts with '2'
                oneStockData.append(oneRow)
    print i,'/',len(stocks)
    dataset.update({os.path.basename(stock)[0:-7]:oneStockData})
    i += 1


Date = 0
Time = 1
Open = 2
High = 3
Low = 4
Close = 5
Volume = 6
OpenInt = 7

print 'Building timeline'
timeLine = []               # List of all different timestamps in order
for eachTimeSeries in dataset.values():
    for onePoint in eachTimeSeries:
        oneTime = dt.strptime(onePoint[Date]+","+onePoint[Time],"%Y-%m-%d,%H:%M:%S")
        if timeLine == None:
            timeLine.append(oneTime)
        elif oneTime not in timeLine:
            timeLine.append(oneTime)
else:
    timeLine = sorted(timeLine)
            

# Dict of index : timestamp
index_TimeLine = {}
print 'Building timeline index'
i=0
for eachTimeStamp in timeLine:
   print i
   index_TimeLine.update({eachTimeStamp:i})
   i+=1
   
for eachTimeSeries in dataset.values():
    for i in range(len(eachTimeSeries)):        
        oneTime = dt.strptime(eachTimeSeries[i][Date]+","+eachTimeSeries[i][Time],"%Y-%m-%d,%H:%M:%S")
        eachTimeSeries[i][Date] = None
        eachTimeSeries[i][Time] = index_TimeLine[oneTime]   # Transform string to index

# Transform the dataset into padas Series
print 'Building formated dataset'
formatDataset = []
times=0
for eachTimeSeries in dataset.values():
    oneStock = []
    print times,'/',len(dataset)
    times+=1
    i = 0
    if len(eachTimeSeries) == 0:
        continue
    for timestamp in range(len(index_TimeLine)):
        if i >= len(eachTimeSeries):
            break
        if timestamp != eachTimeSeries[i][Time]: 
            oneStock.append(np.nan)
        else:
            oneStock.append(np.float32(eachTimeSeries[i][Open]))    # Only load the open price
            i+=1
            
    start_of_valid = 0
    for each in oneStock:
        if np.isnan(each):
            start_of_valid += 1
        else:
            break
        
    end_of_valid = len(oneStock)-1
    while end_of_valid > start_of_valid:
        if np.isnan(oneStock[end_of_valid]):
            end_of_valid -= 1
        else:
            break
    assert not np.isnan(oneStock[start_of_valid])
    assert not np.isnan(oneStock[end_of_valid])
    oneStock_temp = np.array(oneStock[start_of_valid:end_of_valid])
    # Only interpolate between valid data!
    nans,temp = nan_helper(oneStock_temp)
    if len(nans) != 0:
        oneStock_temp[nans]= np.interp(temp(nans), temp(~nans), oneStock_temp[~nans])
        oneStock[start_of_valid:end_of_valid] = oneStock_temp
    oneStock = list(oneStock)
    # Reference: https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    formatDataset.append(oneStock)            


'''
Close to 1: postively correlated
Close to 0: not correlated
Close to -1: negatively correlated
'''




# Dictionary of stock name and their indices  
index_stockname = {}    
i=0
for eachStock in stocks:
   index_stockname.update({i:os.path.basename(eachStock)[0:-7]})         
   i+=1
# Stock name - index dict:
stockname_index=dict([[stock,index] for index,stock in index_stockname.iteritems()])
# Index - stock name dict: index_stockname

# The valid  length of data of each stock
validLength = []
for eachStock in formatDataset:
    length = 0
    for eachPrice in eachStock:
        if not np.isnan(eachPrice):
            length +=1
    validLength.append(length)

if 'regr_output' not in os.listdir(os.getcwd()):
    os.mkdir('regr_output')



# Plot all the data
'''
for each in formatDataset:
    #if each[28595] < 10000:
     each.plot()
'''    
# Which is time consuming

# Plot 50 stocks
'''
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for i in range(50):
    ax.plot(formatDataset[i+123])
    

#ax = formatDataset[i+123].plot()
ax.set_xlabel('Time Stamp')
ax.set_ylabel('Price')
fig.savefig('1.png',dpi=1000)
'''
def plot_and_save(stock):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if type(stock) == str:
        index = stockname_index[stock]
        print index
        ax.plot(formatDataset[index],label=str(stock))
        ax.set_xlabel("Time Stamp")
        ax.set_ylabel("Price")
        ax.legend()
    elif type(index) == list:
        for i in index:
            if type(i) == int or type(i) == np.int:
                ax.plot(formatDataset[i])
            else:
                j = stockname_index[i]
                ax.plot(formatDataset[j])
    else:
        ax.plot(formatDataset[index+1])
    fig.savefig(stock+'.png',dpi=400)


# Where does each stock price data start and end.
print 'Search for valid start and end index'
valid_start_index = []
valid_end_index = []
for i in range(len(formatDataset)):
    print i,'/',len(formatDataset)
    for k in range(len(formatDataset[i])):
        if not np.isnan(formatDataset[i][k]):
            break
    valid_start_index.append(k)
    for l in range(len(formatDataset[i])):
        if not np.isnan(formatDataset[i][-l-1]):
            # end point found
            break
    valid_end_index.append(len(formatDataset[i])-l-1)
    assert len(formatDataset[i])-l-1 > k


def correlation(i,j):
    global formatDataset
    global valid_start_index
    global valid_end_index
    start_point = max(valid_start_index[i],valid_start_index[j])
    end_point = min(valid_end_index[i],valid_end_index[j])
    
    
    if end_point < start_point+800:
    # Data not sufficient !
        return 0    
    #print start_point,end_point
    #assert True not in np.isnan(formatDataset[i][start_point:end_point])
    #assert True not in np.isnan(formatDataset[j][start_point:end_point])
    output= np.corrcoef(formatDataset[i][start_point:end_point],formatDataset[j][start_point:end_point])
    return output[0][1]
    
def compute_corr():
    output = dict()
    for i in range(len(formatDataset)):
        print i,'/',len(formatDataset)
        output.update({index_stockname[i]:dict()})
        for j in range(i+1,len(formatDataset)):
            corr = correlation(i,j)
            if corr > 0.8 or corr < -0.8:
                output[index_stockname[i]].update({index_stockname[j]:corr})
    
    return output


def write_corr(data,threshold):
    #output_file = open('all_correlations_'+str(threshold)+'.csv','wb')
    output_file = open('correlations_'+str(threshold)+'.csv','wb')
    writer = csv.writer(output_file)
    writer.writerow(['stock1','stock2','correlation'])
    for stock in data.iterkeys():
        for anotherStock in data[stock].iterkeys():
            corr = data[stock][anotherStock]
            if corr > threshold or corr < -threshold:
                writer.writerow([stock,anotherStock,corr])
    output_file.close()     
    
# Takes long time to calculate!
# I have done this part, and exported the data to files
'''
print 'Computing correlation'
correlation_dict = compute_corr()
write_corr(correlation_dict,0.8)
write_corr(correlation_dict,0.9)
write_corr(correlation_dict,0.95)
'''

# Short term moving average
def SMA(index,t,s):
    global formatDataset
    temp = 0
    for i in range(1,s+1):
        if not np.isnan(formatDataset[index][t-i+1]):
            temp += formatDataset[index][t-i+1]
        else:
            #s -=1
            raise ValueError('Invalid data!')
            return None
    temp = temp/s
    return temp

# Long term moving average
def LMA(index,t,l):
    global formatDataset
    temp = 0
    for i in range(1,l+1):
        if not np.isnan(formatDataset[index][t-i+1]):   
            temp += formatDataset[index][t-i+1]
        else:
            #l -=1
            raise ValueError('Invalid data!')
            return None
    temp = temp/l
    return temp


def load_correlation_dict():
    correlation_data = dict()
    for stockname in stockname_index.iterkeys():       
        correlation_data.update({stockname:dict()})
    
    #with open('all_correlations_0.9.csv','r') as correlation_file:
    with open('correlations_0.9.csv','r') as correlation_file:
        reader = csv.reader(correlation_file)
        reader.next()
    
        inputLine = reader.next()
        while(len(inputLine) >0):
            correlation_data[inputLine[0]].update({inputLine[1]:inputLine[2]})
            correlation_data[inputLine[1]].update({inputLine[0]:inputLine[2]})
            try: 
                inputLine = reader.next()
            except:
                print 'Finish reading correlation data.'
                break
    return correlation_data

correlation_data = load_correlation_dict()

def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape,fillval)
    out[mask] = np.concatenate(v)
    return out


def plot_dataset(data_set,b_label):
    # Only are able to plot 2D graph
    if type(data_set) == list:
        assert len(data_set[0])==2
    elif type(data_set) == np.ndarray:
        assert data_set.shape[1] == 2
        
    fig, ax = plt.subplots()
    ax.scatter(data_set.transpose()[0],data_set.transpose()[1],c=b_label,marker='s',s=[1])
    
    fig.savefig('dataset.png',dpi=400)



def get_train_test_data(stock,l_max=500,l_ratio=4.0,s_max=30,binary=False):

    index = stockname_index[stock]
    
    l=max(l_max,validLength[index]/float(l_ratio))  
    l = int(l)

    s = max(s_max,l/100)
    if s > 50:
        s = 50

    
    print 'l=',l,'s=',s
# Starting point and ending point:
#starting_point  = len(formatDataset[index])-validLength[index]+l
    end_point = len(formatDataset[index])
    for corrStock,_ in correlation_data[stock].iteritems():
        i=stockname_index[corrStock]
        end_point = min(len(formatDataset[i]),end_point)

# Build the training set
    data_set = []
    for corrStock,corr in correlation_data[stock].iteritems():
        data=[]
        i=stockname_index[corrStock]
        
# Start at where target stock price data is valid
        #starting_point  = len(formatDataset[index])-validLength[index]+l
        starting_point = valid_start_index[index] + l
        while starting_point < end_point:#len(formatDataset[index]):
            if np.isnan(formatDataset[i][starting_point-l]):
# If correlated stock data is not valid
                data.append(np.nan)
                starting_point += 1
                continue
            data.append((LMA(i,starting_point,l)-SMA(i,starting_point,s))*np.float(corr))
            starting_point += 1
        if data != []:
            data_set.append(data)
            
    
    if len(data_set) == 0:
        # If no valid data
        print 'Not enough valid data for stock',stock,'!'
        raise InsufficientData(stock)
    
# Disgard the data if too many values are nan
    ave_nan_percent = 0.0
# Compute average percentage of nan elements for each stock
    for i in range(len(data_set)):
        each = data_set[i]
        length = len(each)
        num_nans = sum(np.isnan(each))
    
    # If more than one-third is nan
        ave_nan_percent+= float(num_nans)/length

    ave_nan_percent /= len(data_set)

# Delete those with too many invalid data
    i=0
    while i<len(data_set):
        each = data_set[i]
        length = len(each)
        num_nans = sum(np.isnan(each))
        if num_nans > length/2.0 and num_nans > length * ave_nan_percent :
            del data_set[i]
        else:
            i+=1
            
    data_set = list(np.array(data_set).transpose())
    i=0
    while i<len(data_set):
        eachTimeSlice = data_set[i]
        if True in np.isnan(eachTimeSlice):
            del data_set[i]
        else:
            i+=1
        
# Assert that all data points in training set are valid        
    for each in data_set:
        assert True not in np.isnan(each)
        assert len(each) == len(data_set[0])
    
    if len(data_set) == 0:
        # If no valid data
        print 'Not enough valid data for stock',stock,'!'
        raise InsufficientData(stock)
    
        

    data_set=np.array(data_set)
    output_length = data_set.shape[0]

# Set invalid data with nan
    data_set = boolean_indexing(data_set)
# Scale data to zero mean and 1 variance
    data_set=preprocessing.scale(data_set)
    

    if binary:
        # Bianry: rise or drop
        b_label = []
        starting_point  = len(formatDataset[index])-validLength[index]+l
        init_price = formatDataset[index][starting_point-1]
        while starting_point <len(formatDataset[index]):
            if formatDataset[index][starting_point] >= formatDataset[index][starting_point-1]:
                b_label.append(1)
            else:
                b_label.append(0)
            starting_point += 1
        b_label = b_label[0:output_length]
    
        # Generate param
        training_size = np.int(0.7 * data_set.shape[0])
        length_of_validation = min(len(b_label),len(data_set))-training_size-1
        
        # Define training data and test data
        training_set = data_set[0:training_size]
        test_set = data_set[training_size+1:training_size+1+ length_of_validation]
        
        b_training_label = b_label[0:training_size]
        b_test_label = b_label[training_size+1:training_size+1+length_of_validation]
        if len(b_training_label) == sum(b_training_label) or sum(b_training_label) == 0:
            print 'Not enough valid data of',stock,'!'
            raise InsufficientData(stock)
            
        if len(b_test_label) == sum(b_test_label) or sum(b_test_label) == 0:
            print 'Not enough valid data of',stock,'!'
            raise InsufficientData(stock)
        return training_set,test_set,b_training_label,b_test_label

    else:
    # Continuous: normalized stock price 
        label = []
        starting_point  = len(formatDataset[index])-validLength[index]+l
        init_price = formatDataset[index][starting_point-1]
        while starting_point <len(formatDataset[index]):
            label.append(formatDataset[index][starting_point]/init_price)
            starting_point += 1
        label = label[0:output_length]
        label = preprocessing.scale(label)
        
        # Generate param
        training_size = np.int(0.7 * data_set.shape[0])
        length_of_validation = min(len(label),len(data_set))-training_size-1

        # Define training data and test data
        training_set = data_set[0:training_size]
        test_set = data_set[training_size+1:training_size+1+ length_of_validation]
               
        training_label = label[0:training_size]
        test_label = label[training_size+1:training_size+1+length_of_validation]
        if len(test_label) ==0:
            print 'Not enough valid data of',stock,'!'
            raise InsufficientData(stock)
        return training_set,test_set,training_label,test_label




    
    
    
    


# Test the binary data and print accuracy for each model
def test_binary(stock):
    
    score = []
    training_set,test_set,b_training_label,b_test_label=get_train_test_data(stock,binary=True)
    
    #clf = tree.DecisionTreeRegressor()
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    #clf.fit(data_set[0:training_size],b_label[0:training_size])
    clf.fit(training_set,b_training_label)
    #print 'Decision Tree:',clf.score(data_set[training_size+1:training_size+1+ length_of_validation], b_label[training_size+1:training_size+1+length_of_validation])
    print 'Decision Tree:',clf.score(test_set,b_test_label)
    score.append(clf.score(test_set,b_test_label))
    
    gnb = GaussianNB()
    #gnb.fit(data_set[0:training_size],b_label[0:training_size])
    gnb.fit(training_set,b_training_label)
    #print 'Naive Bayes, Gaussain:',gnb.score(data_set[training_size+1:training_size+1+ length_of_validation], b_label[training_size+1:training_size+1+length_of_validation])
    print 'Naive Bayes, Gaussain:',gnb.score(test_set,b_test_label)
    score.append(gnb.score(test_set,b_test_label))
    
    bnb = BernoulliNB()
    #bnb.fit(data_set[0:training_size],b_label[0:training_size])
    bnb.fit(training_set,b_training_label)
    #print 'Naive Bayes, Bernoulli:',bnb.score(data_set[training_size+1:training_size+1+ length_of_validation], b_label[training_size+1:training_size+1+length_of_validation])
    print 'Naive Bayes, Bernoulli:',bnb.score(test_set,b_test_label)
    score.append(bnb.score(test_set,b_test_label))
    
    clf_svm = svm.SVC()
    #clf_svm = svm.SVR()
    
    clf_svm.fit(training_set,b_training_label)
    #print 'SVM:',clf_svm.score(data_set[training_size+1:training_size+1+ length_of_validation], b_label[training_size+1:training_size+1+length_of_validation])
    print 'SVM:',clf_svm.score(test_set,b_test_label)
    score.append(clf_svm.score(test_set,b_test_label))
    
    
    #knn = KNeighborsClassifier(n_neighbors=500).fit(training_set,b_training_label)
    #knn.score(test_set,b_test_label)
    return score

# Regression using Kernel regression
def regr(stock,show=False,save=False):
    
    training_set,test_set,training_label,test_label = get_train_test_data(stock,l_max=500,l_ratio=4.0,s_max=5,binary=False)    
    
    # Decision Tree regressor
    '''
    clf = tree.DecisionTreeRegressor()
    #clf = tree.DecisionTreeClassifier(criterion='entropy')
    #clf.fit(data_set[0:training_size],b_label[0:training_size])
    clf.fit(training_set,training_label)
    #print 'Decision Tree:',clf.score(data_set[training_size+1:training_size+1+ length_of_validation], b_label[training_size+1:training_size+1+length_of_validation])
    print 'Decision Tree:',clf.score(test_set,test_label)
    '''
    kernel_rgr = KernelRidge(alpha=1.0,kernel = 'linear')
    kernel_rgr.fit(training_set,training_label)
    kernel_score = kernel_rgr.score(test_set,test_label)
    print 'Kernel Regression:',kernel_score
    
    # Gaussian Process Regressor
    '''
    gpr = GaussianProcessRegressor(alpha = 1e-3,n_restarts_optimizer = 10,normalize_y=True)
    gpr.fit(training_set,training_label)
    
    print "Gaussain Process Regression:",gpr.score(test_set,test_label)
    '''
    
    # SVM with Polynomial model
    '''
    #clf_svm = svm.SVC()
    clf_svm = svm.SVR(kernel='poly', C=1e2, degree=3)
    
    clf_svm.fit(training_set,training_label)
    #print 'SVM:',clf_svm.score(data_set[training_size+1:training_size+1+ length_of_validation], b_label[training_size+1:training_size+1+length_of_validation])
    svm_score = clf_svm.score(test_set,test_label)
    print 'SVM:',svm_score
    '''
    if show:
        fig = plt.figure()  
        ax = fig.add_subplot(1,1,1)
        #ax.plot(clf_svm.predict(test_set),label='SVM')
        #ax.plot(clf.predict(test_set),label = 'DT')
        ax.plot(test_label,label='Actual data')
        ax.plot(kernel_rgr.predict(test_set),label='Kernel Rgr')
        ax.set_xlabel("Relative Time Stamp")
        ax.set_ylabel("Normalized Price")
        ax.legend()
        if save:
            #if kernel_score >=0.3:
            fig.savefig("regr_output\\"+stock+'_regr_output_'+str(round(kernel_score,4))+'.png',dpi=500)
            plt.close('all')
    return kernel_score#,svm_score]
    
# Test the dataset on binary prediction 
# Takes really long time!
'''
print 'Test for binary data:'
#l=1000

i=0
b_scores=[]
#tested_stocks = []
for each in correlation_data.iterkeys():
    if len(correlation_data[each]) != 0:
        print each
        try:
            temp = test_binary(each)
        except InsufficientData:
            continue
        
            # Some stocks will not tested because of lack of valid data
        #tested_stocks.append(each)
        i+=1
        print i
        #if type(temp) == list and len(temp) ==4:
        b_scores.append(temp)
        print
        if len(b_scores) >= 20:
            break
#print np.mean(b_scores)
print np.mean([each for [each,_,_,_] in b_scores])
print np.mean([each for [_,each,_,_] in b_scores])
print np.mean([each for [_,_,each,_] in b_scores])
print np.mean([each for [_,_,_,each] in b_scores])
print
'''


# Test the regression
'''
print 'Test for continuous data:'
    
#l=1000
#s=30
i=0
scores=[]
tested_stocks = []
for each in correlation_data.iterkeys():
    if len(correlation_data[each]) != 0:
        print each
        try:
            temp = regr(each,show=True,save=True)
        except InsufficientData:
            continue
        
            # Some stocks will not tested because of lack of valid data
        tested_stocks.append(each)
        i+=1
        print i
        scores.append(temp)
        print
        #if len(scores) > 30:
            #break

print np.mean(scores)
'''
def draw_dataset_for_ML_binary(stock,save=False):
    print 'Plotting the dataset with binary labels for ML'
    training_set,test_set,b_training_label,b_test_label = get_train_test_data(stock,binary=True)    
    ML_dataset = np.array(list(training_set)+list(test_set))
    ML_label = np.array(list(b_training_label)+list(b_test_label))
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)

    # Only take two random corelated stocks
    num_of_correlated = ML_dataset.shape[1]
    if num_of_correlated == 1:
        print 'Only one correlated stock.'
        ax.scatter(ML_dataset,ML_label)
        ax.set_xlabel('Normalized (LMA-SMA)*corr')
        ax.set_ylabel('Drop or rise')
    if num_of_correlated==2:
        print 'Found 2 correlated stocks.'
        ax.scatter(ML_dataset[:,0],ML_dataset[:,1],c=ML_label)
        ax.set_xlabel('Normalized (LMA-SMA)*corr')
        ax.set_ylabel('Normalized (LMA-SMA)*corr')
    else:
        print 'More than 2 correlated stocks.'
        import random
        a,b,c = random.sample(range(ML_dataset.shape[1]),3)
       # a,b,c=chosen_stocks
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(ML_dataset[:,a],ML_dataset[:,b],ML_dataset[:,c],c=ML_label)

        ax.set_xlabel('Normalized L-S')
        ax.set_ylabel('Normalized L-S')
        ax.set_zlabel('Normalized L-S')

    ax.legend()
    if save:
        fig.savefig(stock+'_binary_ML_dataset_',dpi=400)
    plt.close('all')


def draw_dataset_for_ML_continuous(stock,save=False):
    print 'Plotting the dataset with continuous labels for ML'
    training_set,test_set,training_label,test_label = get_train_test_data(stock,l_max=500,l_ratio=4.0,s_max=5,binary=False)        
    ML_dataset = np.array(list(training_set)+list(test_set))
    ML_label = np.array(list(training_label)+list(test_label))
    fig=plt.figure()
    num_of_plots = len(ML_dataset[0])
    
    for i in range(num_of_plots):
        ax=fig.add_subplot(1,1,1)
        ax.scatter(ML_dataset[:,i],ML_label)
        ax.set_xlabel('Normalized (LMA-SMA)*corr')
        ax.set_ylabel('Normalized price')
        ax.legend()
        if save:
            fig.savefig(stock+'_ML_dataset_'+str(i),dpi=400)
            fig.show()
            plt.close('all')


def draw_LMA_SMA(stock,save=False):
    print 'Drawing price and moving average of stock',stock
    #global l
    #global s
    index = stockname_index[stock]
    l=max(1000,validLength[index]/6.0)
    l = int(l)
    s = max(1,l/150)

    index = stockname_index[stock]
    start = valid_start_index[index]+l
    end = valid_end_index[index]
    if start >= end:
        raise InsufficientData(stock)
    timestamp = []
    lma = []
    sma = []
    for t,price in enumerate(formatDataset[index][start:end]):
        lma.append(LMA(index,start+t,l))
        sma.append(SMA(index,start+t,s))
        timestamp.append(t)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(timestamp,lma,label='LMA')
    ax.plot(timestamp,sma,label='SMA')
    ax.plot(timestamp,formatDataset[index][start:end],label='Price')
    ax.legend()
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Price')
    if save:
        fig.savefig(stock+'_SMA_LMA.png',dpi=300)



# GUI
        
from PIL import Image
from appJar import gui



def press(button):
    global app
    stock_to_draw = app.getEntry("Stock name")
    if button == "Close":
        app.stop()
    elif button == "Draw LMA and SMA":
        draw_LMA_SMA(stock_to_draw,True)
        print "Showing graph"        
        files = os.listdir(os.getcwd())        
        for oneFile in files:
            if oneFile.startswith(stock_to_draw+'_SMA_LMA') and oneFile.endswith('.png'):
                Image.open(oneFile).show()
        app.stop()
        app = show_gui()    
        app.go()
                
    elif button == "Draw continuous data":
        draw_dataset_for_ML_continuous(stock_to_draw,True)
        files = os.listdir(os.getcwd())
        for oneFile in files:
            if oneFile.startswith(stock_to_draw+'_ML_dataset_'):
                Image.open(oneFile).show()
        app.stop()
        app = show_gui()    
        app.go()
        
                
    elif button == "Draw binary data":
        draw_dataset_for_ML_binary(stock_to_draw,True)
        files = os.listdir(os.getcwd())
        for oneFile in files:
            if oneFile.startswith(stock_to_draw+'_binary_ML_dataset_'):
                Image.open(oneFile).show()
        app.stop()
        app = show_gui()    
        app.go()

    elif button == "Predict":
        regr(stock_to_draw,True,True)
        files = os.listdir(os.getcwd()+ "\\regr_output\\")
        for oneFile in files:
            if oneFile.startswith(stock_to_draw+'_regr_output_'):
                Image.open(os.getcwd()+ "\\regr_output\\"+oneFile).show()
        app.stop()
        app = show_gui()    
        app.go()

    elif button == "Draw original":
        plot_and_save(stock_to_draw)
        files = os.listdir(os.getcwd())
        for oneFile in files:
            if oneFile == stock_to_draw+".png":
                Image.open(oneFile).show()
        app.stop()
        app = show_gui()    
        app.go()

        

# GUI will exit after drawing a graph to show it.
# Run these lines to open GUI       
def show_gui():
    app = gui()
    app.addLabel("title","Correlation and Momentum")
    app.setLabelBg("title", "red")
    app.addLabelEntry("Stock name")
    app.addButtons(["Draw original","Draw LMA and SMA","Draw continuous data","Draw binary data","Predict", "Close"], press)        

    return app


app = show_gui()    
app.go()


'''
# Pipeline:
Dataset -> preprocess -> SMA and LMA -> normalize -> learn and predict
'''

