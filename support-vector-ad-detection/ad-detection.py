
import numpy as np
import pandas as pd



dataFile='/Users/rashmisharma/Documents/Spring 2017/machine-learning/how-to-Pluralsight/classification/support-vector-ad-detection/ad-dataset/ad.data'
data=pd.read_csv(dataFile,sep=",",header=None, low_memory=False)

def toNum(cell):
    try:
        return np.float(cell)
    except:
        return np.nan

def seriestoNum(series):
    return series.apply(toNum)

train_data= data.iloc[0:,0:-1].apply(seriestoNum)

train_data=train_data.dropna()

def toLabel(value):
    if value=='ad.':
        return 1
    else:
        return 0

train_labels= data.iloc[train_data.index,-1].apply(toLabel)


from sklearn.svm  import LinearSVC
clf=LinearSVC()
clf.fit(train_data[100:2300],train_labels[100:2300])


clf.predict(train_data.iloc[0].reshape(1,-1))
