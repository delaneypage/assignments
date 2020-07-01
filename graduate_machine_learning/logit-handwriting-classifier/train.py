#! /bin/env python3

# Loading packages

import pickle
import numpy as np
from scipy import ndimage
import skimage
import pandas as pd

from sklearn.linear_model import LogisticRegression


def load_pkl(fname):
	with open(fname,'rb') as f:
		return pickle.load(f)

def save_pkl(fname,obj):
	with open(fname,'wb') as f:
		pickle.dump(obj,f)

# Loading in the train data
train_data = load_pkl('train_data.pkl')

# Loading in the labels
train_labels = np.load('finalLabelsTrain.npy')

# When attempting to only classify a and b, looking only at reduced set.
ab_train_data = train_data[np.logical_or((train_labels == 1),(train_labels == 2))]
ab_train_labels = train_labels[np.logical_or((train_labels == 1),(train_labels == 2))]

rot_list = []
j = 0

for i in range(1600):
    if(np.shape(ab_train_data[i])[0] < np.shape(ab_train_data[i])[1]):
        rot_list.append(j)
    j = j + 1

rot_list = [241,242,243,244,245,246,247,250,
           251,252,253,254,255,256,257,258,259,
           500,501,502,503,504,505,506,507,508,509,
           510,511,512,513,514,515,516,517,518,519]

# Some things should just be thrown out
trash_list = [240,248,249,960]

# Rotating the above images by 270 degrees, seems to be the only way things went wrong
for index in rot_list:
    img = (ab_train_data[index])
    lx, ly = img.shape
    rot_img = ndimage.rotate(img, 270)
    ab_train_data[index] = rot_img

# For every image we resize to (50, orignal length)
for i in range(1600):
    ab_train_data[i] = skimage.transform.resize(np.asarray(ab_train_data[i]), (50,len(np.array(ab_train_data[i])[0,:])))
    
    
def get_measurements(img):
    
    yaxis = pd.Series([np.count_nonzero(img[i,:])*[i+1] 
           for i in np.arange(np.shape(img)[0])], name='yaxis').explode()
    xaxis = pd.Series([np.count_nonzero(img[:,i])*[i+1] 
           for i in np.arange(np.shape(img)[1])], name='xaxis').explode()   
    
    yrange = (np.max(yaxis) - np.min(yaxis))
    xrange = (np.max(xaxis) - np.min(xaxis))   
    
    rangediff = abs(yrange - xrange)
    
    return yaxis, xaxis, rangediff

yaxis_lis=[]
xaxis_lis=[]
rangediff_lis=[]
yargmax=[]
xargmax=[]
xaxismed=[]

for img in ab_train_data:
    
    yaxis, xaxis, rangediff = get_measurements(img)
        
    yarg = yaxis.reset_index().groupby('yaxis').count().values.argmax()
    xarg = xaxis.reset_index().groupby('xaxis').count().values.argmax()
    
    yargmax.append(yarg)
    xargmax.append(xarg)
    
    yaxis_lis.append(yaxis)
    xaxis_lis.append(xaxis)
    xaxismed.append(xaxis.median())
    rangediff_lis.append(rangediff)
    
X = np.array([np.array(x) for x in list(zip(yargmax, xaxismed, rangediff_lis, xargmax))])
y = ab_train_labels

def train_model(X, y):
    
    print("Training model...")
    
    clf = LogisticRegression(solver='liblinear', penalty='l1').fit(X, y)
    
    filename = 'logreg.sav'
    
    pickle.dump(clf, open(filename, 'wb'))
    
    print("Model training complete. Model saved as ", filename)
    
train_model(X, y)
