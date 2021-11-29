import matplotlib
import pandas as pd
from pandas import read_csv
import matplotlib.pylab as plt
import numpy as np
import scipy as sp
import os


def isGoodForDriving(x):
    if(x<0.08):
        return 1
    return 0

fields = ['name','timestamp','rssiOne','locationStatus']
fields2 = ['name','timestamp','rssiTwo','locationStatus']
series = read_csv('data/RSSI_dataset/filtered_rssi.csv',header=0, index_col=None, skipinitialspace=True,usecols=fields)
series3 = read_csv('data/RSSI_dataset/filtered_rssi.csv',header=0, index_col=None, skipinitialspace=True,usecols=fields2)
series2 = 'data/data/clean_tac/'

series4 = 'data/dataset_fog_release/dataset/'
listOfFileNames = []

for filename in os.listdir(series4):
    listOfFileNames.append(filename)

listOfDataFrames = []
for filename in listOfFileNames:
    listOfDataFrames.append(read_csv(series4+filename, header=None, sep=" "))

series4 = pd.concat(listOfDataFrames)
series4.columns = ["time_of_sample", "ankleHFA", "ankleV", "ankleHL", "upperLegHFA", "upperLegV", "upperLegHL", "trunkAccelerationHFA", "trunkAccelerationV", "trunkAccelerationHL", "Annotation"]
csv_file_list = ["DK3500_clean_TAC.csv", "JR8022_clean_TAC.csv","SA0297_clean_TAC.csv","BU4707_clean_TAC.csv","HV0618_clean_TAC.csv","SF3079_clean_TAC.csv","MJ8002_clean_TAC.csv","CC6740_clean_TAC.csv","PC6771_clean_TAC.csv","MC7070_clean_TAC.csv","DC6359_clean_TAC.csv","BK7610_clean_TAC.csv","JB3156_clean_TAC.csv"]

listOfDataFrames = []
for filename in csv_file_list:
    listOfDataFrames.append(read_csv(series2+filename))


series2 = pd.concat(listOfDataFrames)

colDrunk = [isGoodForDriving(x) for x in series2['TAC_Reading']]
series2['Target'] = colDrunk
series = series[['name','timestamp','rssiOne', 'locationStatus']]
series3 = series3[['name','timestamp','rssiTwo', 'locationStatus']]
print(series.head())
print(series2.head())
print(series3.head())
print(series4.head())
#series.columns = ['rssiOne','locationStatus']
#series3.columns = ['rssiTwo','locationStatus']
# #series3 =series3[['timestamp','locationStatus','rssiTwo']]
# print(series.head())

#series.plot()
#plt.show()
#series2.plot(x = "timestamp",y = "TAC_Reading" )
#plt.show()
#series3.plot()
#plt.show()


index1 = series.index
#index2 = series2.index
# index3 = series3.index
index4 = series4.index

numbers_of_rows1 = len(index1)
#numbers_of_rows2 = len(index2)
# numbers_of_rows3 = len(index3)
numbers_of_rows4 = len(index4)

print(numbers_of_rows1)
#print(numbers_of_rows2)
# print(numbers_of_rows3)
print(numbers_of_rows4)

#series4 = series4[:-1911000]


##Preprocessing





NameVector = series['name']
ListOfNames = []
for value in NameVector.values:
    ListOfNames.append(value)

ListOfNames = list(set(ListOfNames))
for name in ListOfNames:
    print(name)

ListOfDataFramesOfSeries = []
grouped = series.groupby(series.name)

for name in ListOfNames:
    ListOfDataFramesOfSeries.append(grouped.get_group(name))

for df in ListOfDataFramesOfSeries:
    print(df.head())

print(len(ListOfNames))
print(len(ListOfDataFramesOfSeries))

newDataFrames = []
ListOfData = []

currNum = 0
for df in ListOfDataFramesOfSeries:
    ListOfData.clear()
    index = df.index
    numOfRows = len(index) - 1
    #print(numOfRows+1)
    iter = 0
    Timestamps = df['timestamp'].values
    for row in df.itertuples():
        if(iter<numOfRows):
            NumOfIter = int(((Timestamps[iter+1]-Timestamps[iter])/100)+1)
            iter +=1
            for x in range(1,NumOfIter+1):
                ListOfData.append([row.name,currNum,row.rssiOne,row.locationStatus])
                currNum += 100
        else:
                ListOfData.append([row.name,currNum,row.rssiOne,row.locationStatus])
                currNum += 100
    newDataFrames.append(pd.DataFrame(ListOfData,columns=['name', 'timestamp', 'rssiOne', 'locationStatus']))
    currNum = 0
iter = 1
for dt in newDataFrames:
    # print(dt.head())
    print('DataFrame number'+str(iter))
    # print("#Timestamp")
    # dtType = dt['timestamp'].tolist()
    # print(np.sum(np.isnan(dtType)))
    # print("#rssiOne")
    # dtType = dt['rssiOne'].tolist()
    # print(np.sum(np.isnan(dtType)))
    iter +=1
    print(dt.isnull().values.any())
    
#I dont get any true .isnull().values.any() returns true if i have any missing values in data Frame


NameVector = series3['name']
ListOfNames2 = []
for value in NameVector.values:
    ListOfNames2.append(value)

ListOfNames2 = list(set(ListOfNames2))
for name in ListOfNames2:
    print(name)

ListOfDataFramesOfSeries2 = []
grouped = series3.groupby(series3.name)

for name in ListOfNames:
    ListOfDataFramesOfSeries2.append(grouped.get_group(name))

for df in ListOfDataFramesOfSeries2:
    print(df.head())

print(len(ListOfNames2))
print(len(ListOfDataFramesOfSeries2))

newDataFrames2 = []
ListOfData2 = []

currNum2 = 0
for df in ListOfDataFramesOfSeries2:
    ListOfData2.clear()
    index = df.index
    numOfRows = len(index) - 1
    #print(numOfRows+1)
    iter = 0
    Timestamps = df['timestamp'].values
    for row in df.itertuples():
        if(iter<numOfRows):
            NumOfIter = int(((Timestamps[iter+1]-Timestamps[iter])/100)+1)
            iter +=1
            for x in range(1,NumOfIter+1):
                ListOfData2.append([row.name,currNum2,row.rssiTwo,row.locationStatus])
                currNum2 += 100
        else:
                ListOfData2.append([row.name,currNum2,row.rssiTwo,row.locationStatus])
                currNum2 += 100
    newDataFrames2.append(pd.DataFrame(ListOfData2,columns=['name', 'timestamp', 'rssiTwo', 'locationStatus']))
    currNum2 = 0
iter = 1
for dt in newDataFrames2:
    # print(dt.head())
    print('DataFrame number'+str(iter))
    #print("#Timestamp")
    #dtType = dt['timestamp'].tolist()
    # print(np.sum(np.isnan(dtType)))
    # print("#rssiOne")
    # dtType = dt['rssiOne'].tolist()
    # print(np.sum(np.isnan(dtType)))
    iter +=1
    print(dt.isnull().values.any())
    
#I dont get any true .isnull().values.any() returns true if i have any missing values in data Frame

# Test = newDataFrames[0]
# Test['locationStatus'].replace({"OUTSIDE":3, "IN_VESTIBULE":2, "INSIDE":1}, inplace=True)
# Test1 = Test[['timestamp', 'rssiOne']]
# Test2 = Test[['locationStatus']]
# TestA = newDataFrames2[3]
# TestA['locationStatus'].replace({"OUTSIDE":3, "IN_VESTIBULE":2, "INSIDE":1}, inplace=True)
# TestA1 = TestA[['timestamp', 'rssiTwo']]
# TestA2 = TestA[['locationStatus']]

# TestA = newDataFrames2[0]
# TestA['locationStatus'].replace({"OUTSIDE":3, "IN_VESTIBULE":2, "INSIDE":1}, inplace=True)
# TestA1 = TestA[['timestamp', 'rssiTwo']]
# TestA2 = TestA[['locationStatus']]


# Test = series4.drop('Annotation',axis=1)
# TestA = series4[['Annotation']]
from sklearn.model_selection import train_test_split

# test1np, test2np =  train_test_split(Test, test_size=0.2)
# testA1np, testA2np =  train_test_split(TestA, test_size=0.2)

# labels, count = np.unique(Testnp, return_counts = True)
# # print(labels, count)
# test1np = Test1.to_numpy()
# test2np = Test2.to_numpy()
# testA1np = TestA1.to_numpy()
# testA2np = TestA2.to_numpy()

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# series['locationStatus'].replace({"OUTSIDE":3, "IN_VESTIBULE":2, "INSIDE":1}, inplace=True)
# series3['locationStatus'].replace({"OUTSIDE":3, "IN_VESTIBULE":2, "INSIDE":1}, inplace=True)
# classifier = DummyClassifier(strategy='prior')
# # for df in newDataFrames:
# #     df['locationStatus'].replace({"OUTSIDE":3, "IN_VESTIBULE":2, "INSIDE":1}, inplace=True)
# #     classifier.fit(df[['timestamp', 'rssiOne']].to_numpy(), df[['locationStatus']].to_numpy().ravel())
# # classifier.fit(test1np, test2np)
# ts = series3[['timestamp', 'rssiTwo']]
# classifier.fit(series[['timestamp', 'rssiOne']].to_numpy(), series[['locationStatus']].to_numpy())
# val = classifier.predict(ts)
# numofPred = accuracy_score(series3[['locationStatus']].to_numpy(), val, normalize=False)
# score = accuracy_score(series3[['locationStatus']].to_numpy(), val)
# #score = classifier.score(series3[['timestamp', 'rssiOne']].to_numpy(), series[["locationStatus"]].to_numpy())
# print(score)

# print(val)
# print(numofPred)

# #test jezeli nauczam calym framem
# series['locationStatus'].replace({"OUTSIDE":3, "IN_VESTIBULE":2, "INSIDE":1}, inplace=True)
# series3['locationStatus'].replace({"OUTSIDE":3, "IN_VESTIBULE":2, "INSIDE":1}, inplace=True)
# series['timestamp'] = series['timestamp'].astype(np.double)
# series['rssiOne'] = series['rssiOne'].astype(np.double)
# test1np = series[['timestamp', 'rssiOne']]
# test2np = series[['locationStatus']].to_numpy()
# series3['timestamp'] = series3['timestamp'].astype(np.double)
# series3['rssiTwo'] = series3['rssiTwo'].astype(np.double)
# testA1np = series3[['timestamp', 'rssiTwo']]
# testA2np = series3[['locationStatus']].to_numpy()


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)

# TestA = newDataFrames2[0]
# TestA['locationStatus'].replace({"OUTSIDE":3, "IN_VESTIBULE":2, "INSIDE":1}, inplace=True)
# TestA1 = TestA[['timestamp', 'rssiTwo']]
# TestA2 = TestA[['locationStatus']]

# for df in newDataFrames:
#     #df['locationStatus'].replace({"OUTSIDE":3, "IN_VESTIBULE":2, "INSIDE":1}, inplace=True)
#     classifier.fit(df[['timestamp', 'rssiOne']].to_numpy(), df[['locationStatus']].to_numpy().ravel())


#print(TestA['locationStatus'].value_counts())
#series['locationStatus'].replace({"OUTSIDE":3, "IN_VESTIBULE":2, "INSIDE":1}, inplace=True)
# print(series['locationStatus'].value_counts())
# y_pred = classifier.predict(series[['timestamp', 'rssiOne']].to_numpy())
# score = accuracy_score(series[['locationStatus']].to_numpy(), y_pred)
# scoreAmount = accuracy_score(series[['locationStatus']].to_numpy().ravel(), y_pred,normalize=False)
# print(y_pred)
# print(scoreAmount)
# print(score)
from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree
# from sktime.utils.slope_and_trend import _slope

# print(series4['Annotation'].value_counts())
# time_series_tree = DecisionTreeClassifier()
# time_series_tree.fit(test1np.to_numpy(), testA1np.to_numpy().ravel())

# y_pred = time_series_tree.predict(test2np.to_numpy())
# score = accuracy_score(testA2np.to_numpy().ravel(), y_pred)
# scoreAmount = accuracy_score(testA2np.to_numpy().ravel(), y_pred,normalize=False)
# print(y_pred)
# print(scoreAmount)
# print(score)


#time_series_tree.score(test2np.to_numpy(), testA2np.to_numpy().ravel())
# for df in newDataFrames:
#     df['locationStatus'].replace({"OUTSIDE":3, "IN_VESTIBULE":2, "INSIDE":1}, inplace=True)
#     time_series_tree.fit(df[['timestamp', 'rssiOne']].to_numpy(), df[['locationStatus']].to_numpy().ravel())
#     y_pred = time_series_tree.predict(TestA1.to_numpy())
#     score = accuracy_score(TestA2.to_numpy().ravel(), y_pred)
#     scoreAmount = accuracy_score(TestA2.to_numpy().ravel(), y_pred,normalize=False)
#     print(y_pred)
#     print(scoreAmount)
#     print(score)


#clf = time_series_tree.fit(test1np.to_numpy(), test2np.to_numpy().ravel())
#tree.plot_tree(clf)
#print(time_series_tree.score(testA1np.to_numpy(), testA2np.to_numpy().ravel()))
#print(time_series_tree.score(series3[['timestamp', 'rssiTwo']].to_numpy(), series3[['locationStatus']].to_numpy().ravel()))
from sktime.datatypes._panel._convert import from_2d_array_to_nested

# test1np = from_2d_array_to_nested(test1np)
# testA1np = from_2d_array_to_nested(testA1np)


# Test = series4.drop('Annotation',axis=1)
# TestA = series4[['Annotation']]



# series['locationStatus'].replace({"OUTSIDE":3, "IN_VESTIBULE":2, "INSIDE":1}, inplace=True)
# print("debug zamiana DF na nested DF")
# series = series[['timestamp', 'rssiOne','locationStatus']].astype('double')
# Test = from_2d_array_to_nested(series)


from sklearn.model_selection import train_test_split
# print("debug podział")
# test1np, testA1np =  train_test_split(Test, test_size=0.5)
# test2np, testA2np =  train_test_split(series[['locationStatus']], test_size=0.5)

#Works dont need nested DataFrame
from sklearn.neighbors import KNeighborsClassifier


#Doesnt work stuck on predict
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

from sktime.classification.compose import ComposableTimeSeriesForestClassifier


from sktime.classification.dictionary_based import BOSSEnsemble

from sktime.classification.interval_based import RandomIntervalSpectralForest

from sktime.classification.shapelet_based import ShapeletTransformClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB

for df in newDataFrames:
    df['locationStatus'].replace({"OUTSIDE":3, "IN_VESTIBULE":2, "INSIDE":1}, inplace=True)

for df in newDataFrames2:
    df['locationStatus'].replace({"OUTSIDE":3, "IN_VESTIBULE":2, "INSIDE":1}, inplace=True)
import time
clf = ComposableTimeSeriesForestClassifier()
clf = KNeighborsClassifier()
clf = Perceptron()
clf = BernoulliNB()
#clf = DecisionTreeClassifier()
series['locationStatus'].replace({"OUTSIDE":3, "IN_VESTIBULE":2, "INSIDE":1}, inplace=True)
Classes = np.unique(series['locationStatus'].to_numpy())
iterator = 0
for df in newDataFrames:
    
    df2 = newDataFrames2[iterator]
    name = df['name'].values[0]
    df = df[['timestamp', 'rssiOne', 'locationStatus']].astype('double') 
    df2 = df2[['timestamp', 'rssiTwo', 'locationStatus']].astype('double')
    dfClass = df[['locationStatus']]
    df2Class = df2[['locationStatus']]
    df = df.drop('locationStatus',axis=1)
    df2 = df2.drop('locationStatus', axis=1)
    # t = time.process_time()
    #TrainSet = from_2d_array_to_nested(df)
    TrainSet = df
    # print(time.process_time()-t)
    # t = time.process_time()
    # TestSet = from_2d_array_to_nested(df2)
    # TestSet = df2
    # print(time.process_time()-t)
    # t = time.process_time()
    clf.partial_fit(TrainSet, dfClass.to_numpy().ravel(),Classes)
    # print(time.process_time()-t)
    # t = time.process_time()
    # y_pred = clf.predict(TestSet)
    # print(time.process_time()-t)
    # score = accuracy_score(df2Class.to_numpy().ravel(), y_pred)
    # scoreAmount = accuracy_score(df2Class.to_numpy().ravel(), y_pred,normalize=False)
    # print('wyniki dla ',name,' nr iteratora ', str(iterator))
    # print(y_pred)
    # print(scoreAmount)
    # print(df2Class.value_counts())
    # print(score)
    # iterator+=1

for df2 in newDataFrames2:
    df2 = df2[['timestamp', 'rssiTwo', 'locationStatus']].astype('double')
    df2Class = df2[['locationStatus']]
    df2 = df2.drop('locationStatus', axis=1)
    
    TestSet = from_2d_array_to_nested(df2)
    TestSet = df2
    t = time.process_time()
    y_pred = clf.predict(TestSet)
    print(time.process_time()-t)
    score = accuracy_score(df2Class.to_numpy().ravel(), y_pred)
    scoreAmount = accuracy_score(df2Class.to_numpy().ravel(), y_pred,normalize=False)
    print('wyniki dla ',name,' nr iteratora ', str(iterator))
    print(y_pred)
    print(scoreAmount)
    print(df2Class.value_counts())
    print(score)
    iterator+=1


# knn = KNeighborsTimeSeriesClassifier()  
# print("debug knn")
# knn.fit(test1np, test2np.to_numpy().ravel())
# print("debug knn score")
# y_predict = knn.predict(testA1np)

# score =knn.score(testA1np, testA2np.to_numpy().ravel())
# print(y_predict)
# print(score)
# from sktime.classification.interval_based import RandomIntervalSpectralForest

# rise = RandomIntervalSpectralForest(n_estimators=10)
# rise.fit(test1np, test2np.ravel())
# score =rise.score(testA1np, testA2np.ravel())
# print(score)
# plt.show()

# from pyts.classification import LearningShapelets

# clf = LearningShapelets(shapelet_scale=5)
# clf.fit(test1np, test2np.ravel())
# print(clf.score(testA1np, testA2np.ravel()))
