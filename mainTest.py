import matplotlib
from matplotlib.pyplot import plot
import pandas as pd
from pandas import read_csv
import matplotlib.pylab as plt
import numpy as np
import scipy as sp
import os
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from sktime.datatypes._panel._convert import from_nested_to_2d_array

from load_datasets import load_data_Meat
from load_datasets import load_data_Earthquakes
from load_datasets import load_data_Strawberry
from load_datasets import load_data_FordA
from load_datasets import load_data_Crop
# train_x, train_y = load_from_tsfile_to_dataframe(
#     "data/Meat/Meat_TRAIN.ts"
# )
# test_x, test_y = load_from_tsfile_to_dataframe(
#     "data/Meat/Meat_TEST.ts"
# )
# train_x, train_y = load_from_tsfile_to_dataframe(
#     "data/FaceDetection/FaceDetection_TRAIN.ts"
# )
# test_x, test_y = load_from_tsfile_to_dataframe(
#     "data/FaceDetection/FaceDetection_TEST.ts"
# )
from plot_classification_results import plot_strawberry_results
plot_strawberry_results()
strawberryListTrain, strawberryListTest = load_data_Earthquakes()

train_x, train_y = strawberryListTrain

# train_x = strawberryListTrain[0]
# train_y = strawberryListTrain[1]
test_x, test_y = strawberryListTest

#test_y = strawberryListTest[1]
print(train_x.head())
print(from_nested_to_2d_array(train_x).head())
# print(train_y)
# print(test_x.head())
# print(test_y)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

#Doesnt work stuck on predict
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

from sktime.classification.compose import ComposableTimeSeriesForestClassifier

from sktime.classification.dictionary_based import IndividualBOSS

#Time series Forrest
from sklearn.ensemble import RandomForestClassifier

from sktime.classification.interval_based import RandomIntervalSpectralForest

#Doesnt work stuck on fit probobly beacuse i dont install full sktime and wont beacuse of unsuported mac os on brew
from sktime.classification.shapelet_based import ShapeletTransformClassifier

from sklearn.tree import DecisionTreeClassifier

from pyts.classification import SAXVSM

from pyts.classification import BOSSVS


clf = RandomForestClassifier()
#clf = KNeighborsClassifier()
#clf = KNeighborsTimeSeriesClassifier()
print(train_x.isnull().values.any())

print(test_x.isnull().values.any())
from plot_datasets import plotCrop
plotCrop()

# train_x_2 = from_nested_to_2d_array(train_x) 
# test_x_2 = from_nested_to_2d_array(test_x)


# # print(train_x_2.head())
# clf.fit(train_x_2, train_y)

# y_pred = clf.predict(test_x_2)

# score = accuracy_score(test_y.ravel(), y_pred)
# scoreAmount = accuracy_score(test_y.ravel(), y_pred,normalize=False)

# print(score)
# print(scoreAmount)
# train_x = from_nested_to_2d_array(train_x) 
# test_x = from_nested_to_2d_array(test_x)
# print(train_x.head())
from classify_datasets import KNClassification
from classify_datasets import TSSClasifier
from classify_datasets import RTSSClasifier
from classify_datasets import BOSSClasifier
from classify_datasets import SAXClasifier

# strawberrydf = pd.concat([test_x, train_x])
# print(strawberrydf.isnull().values.any())
# strawberrydf.head()

# classdf = np.concatenate([train_y, test_y])
# labels, counts = np.unique(classdf, return_counts=True)
# print(labels, counts)
# plt.bar(labels,counts)

# #number of classes in given timestamps
# # fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))
# # for label in labels:
# #     strawberrydf.loc[classdf == label, "dim_0"].iloc[0].plot(ax=ax, label=label)
# # plt.legend()
# # ax.set(title="Example time series", xlabel="Time")

# #numbers of instancef of current labels                                 
# for label in labels[:2]:
#     fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))
#     for instance in strawberrydf.loc[classdf == label, "dim_0"]:
#         ax.plot(instance)
#     ax.set(title=f"Instances of {label}")

# plt.show()
# predict,score, scoreAmount, result_time = KNClassification(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)
# predict,score, scoreAmount, result_time = TSSClasifier(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)

# predict,score, scoreAmount, result_time = RTSSClasifier(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)
# predict,score, scoreAmount, result_time = BOSSClasifier(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)
# predict,score, scoreAmount, result_time = SAXClasifier(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)
# strawberryListTrain, strawberryListTest = load_data_Meat()

# train_x, train_y = strawberryListTrain
# test_x, test_y = strawberryListTest

# predict,score, scoreAmount, result_time = KNClassification(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)
# predict,score, scoreAmount, result_time = TSSClasifier(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)

# predict,score, scoreAmount, result_time = RTSSClasifier(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)
# predict,score, scoreAmount, result_time = BOSSClasifier(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)
# predict,score, scoreAmount, result_time = SAXClasifier(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)
# strawberryListTrain, strawberryListTest = load_data_Earthquakes()

# train_x, train_y = strawberryListTrain
# test_x, test_y = strawberryListTest

# predict,score, scoreAmount, result_time = KNClassification(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)
# predict,score, scoreAmount, result_time = TSSClasifier(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)

# predict,score, scoreAmount, result_time = RTSSClasifier(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)
# predict,score, scoreAmount, result_time = BOSSClasifier(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)
# predict,score, scoreAmount, result_time = SAXClasifier(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)
# strawberryListTrain, strawberryListTest = load_data_Strawberry()

# train_x, train_y = strawberryListTrain
# test_x, test_y = strawberryListTest

# predict,score, scoreAmount, result_time = KNClassification(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)
# predict,score, scoreAmount, result_time = TSSClasifier(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)

# predict,score, scoreAmount, result_time = RTSSClasifier(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)
# predict,score, scoreAmount, result_time = BOSSClasifier(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)
# predict,score, scoreAmount, result_time = SAXClasifier(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)
# strawberryListTrain, strawberryListTest = load_data_FordA()

# train_x, train_y = strawberryListTrain
# test_x, test_y = strawberryListTest

# predict,score, scoreAmount, result_time = KNClassification(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)
# predict,score, scoreAmount, result_time = TSSClasifier(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)

# predict,score, scoreAmount, result_time = RTSSClasifier(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)
# predict,score, scoreAmount, result_time = BOSSClasifier(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)
# predict,score, scoreAmount, result_time = SAXClasifier(train_x, train_y, test_x, test_y)
# print(predict)
# print(score)
# print(scoreAmount)
# print(result_time)