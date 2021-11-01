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



#x = series['timestamp'].tolist()
#save numeric columns value in list variable to check whether we have any nullable values in it
y = series['rssiOne'].tolist()
x = series2['TAC_Reading'].tolist()
z = series3['rssiTwo'].tolist()
a = series4['ankleHFA'].tolist()


print(np.sum(np.isnan(y)))
#first series print 0 it means that i dont have any missing values in this series

print(np.sum(np.isnan(x)))
#second series print 0 it means that i dont have any missing values in this series

print(np.sum(np.isnan(z)))
#third series print 0 it means that i dont have any missing values in this series

print(np.sum(np.isnan(a)))
#fourth series print 0 it means that i dont have any missing valyes in this series

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

Test = newDataFrames[0]
Test['locationStatus'].replace({"OUTSIDE":3, "IN_VESTIBULE":2, "INSIDE":1}, inplace=True)
Test1 = Test[['timestamp', 'rssiOne']]
Test2 = Test[['locationStatus']]
# TestA = newDataFrames2[3]
# TestA['locationStatus'].replace({"OUTSIDE":3, "IN_VESTIBULE":2, "INSIDE":1}, inplace=True)
# TestA1 = TestA[['timestamp', 'rssiTwo']]
# TestA2 = TestA[['locationStatus']]

TestA = newDataFrames2[0]
TestA['locationStatus'].replace({"OUTSIDE":3, "IN_VESTIBULE":2, "INSIDE":1}, inplace=True)
TestA1 = TestA[['timestamp', 'rssiTwo']]
TestA2 = TestA[['locationStatus']]


# Test = series4.drop('Annotation',axis=1)
# TestA = series4[['Annotation']]
from sklearn.model_selection import train_test_split

test1np, testA1np =  train_test_split(Test, test_size=0.2)
test2np, testA2np =  train_test_split(TestA, test_size=0.2)
#Testnp = Test2.to_numpy()
#labels, count = np.unique(Testnp, return_counts = True)
#print(labels, count)
# test1np = Test1.to_numpy()
# test2np = Test2.to_numpy()
# testA1np = TestA1.to_numpy()
# testA2np = TestA2.to_numpy()

#from sklearn.dummy import DummyClassifier

#classifier = DummyClassifier(strategy='prior')
#classifier.fit(test1np, test2np)
#score = classifier.score(testA1np, testA2np)
#print(score)

#test jezeli nauczam calym framem
series['locationStatus'].replace({"OUTSIDE":3, "IN_VESTIBULE":2, "INSIDE":1}, inplace=True)
series3['locationStatus'].replace({"OUTSIDE":3, "IN_VESTIBULE":2, "INSIDE":1}, inplace=True)
series['timestamp'] = series['timestamp'].astype(np.double)
series['rssiOne'] = series['rssiOne'].astype(np.double)
test1np = series[['timestamp', 'rssiOne']]
test2np = series[['locationStatus']].to_numpy()
series3['timestamp'] = series3['timestamp'].astype(np.double)
series3['rssiTwo'] = series3['rssiTwo'].astype(np.double)
testA1np = series3[['timestamp', 'rssiTwo']]
testA2np = series3[['locationStatus']].to_numpy()


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# classifier = RandomForestClassifier(n_estimators=100)
# classifier.fit(test1np.to_numpy(), test2np.to_numpy().ravel())
# y_pred = classifier.predict(testA1np.to_numpy())
# score = accuracy_score(testA2np.to_numpy(), y_pred)
# print(score)

from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor

from sklearn.metrics import accuracy_score

# from sklearn.pipeline import Pipeline
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree
# from sktime.utils.slope_and_trend import _slope

# time_series_tree = DecisionTreeClassifier()
# clf = time_series_tree.fit(test1np.to_numpy(), test2np.to_numpy().ravel())
# tree.plot_tree(clf)
# print(time_series_tree.score(testA1np.to_numpy(), testA2np.to_numpy().ravel()))

from sktime.datatypes._panel._convert import from_2d_array_to_nested

test1np = from_2d_array_to_nested(test1np)
testA1np = from_2d_array_to_nested(testA1np)



from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

knn = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw")

knn.fit(test1np, test2np.ravel())
score =knn.score(testA1np, testA2np.ravel())
print(score)
from sktime.classification.interval_based import RandomIntervalSpectralForest

# rise = RandomIntervalSpectralForest(n_estimators=10)
# rise.fit(test1np, test2np.ravel())
# score =rise.score(testA1np, testA2np.ravel())
# print(score)
# plt.show()
# from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor

# transformer = TSFreshFeatureExtractor(default_fc_parameters="minimal")
# extracted_features = transformer.fit_transform(X_train)
# extracted_features.head()
# from pyts.classification import LearningShapelets

# clf = LearningShapelets(shapelet_scale=5)
# clf.fit(test1np, test2np.ravel())
# print(clf.score(testA1np, testA2np.ravel()))
