#! /bin/env python3

# Loading packages
import pickle
import numpy as np
import skimage
import pandas as pd

def load_pkl(fname):
	with open(fname,'rb') as f:
		return pickle.load(f)

def save_pkl(fname,obj):
	with open(fname,'wb') as f:
		pickle.dump(obj,f)
    
print("Loading training data...")    

# Loading in the train data

train_data = load_pkl('train_data.pkl') # === INSERT TEST DATA PATH HERE === #

# Loading in the labels

## ==== Comment out if loading in labels ==== ## 

# train_labels = np.load('finalLabelsTrain.npy')

# When attempting to only classify a and b, looking only at reduced set.

## ==== Comment out if loading in all letters to only test a, b ==== ## 

# train_data = train_data[np.logical_or((train_labels == 1),(train_labels == 2))]
# train_labels = train_labels[np.logical_or((train_labels == 1),(train_labels == 2))]

# For every image we resize to (50, orignal length)
for i in np.arange(len(train_data)):
    train_data[i] = skimage.transform.resize(np.asarray(train_data[i]), (50,len(np.array(train_data[i])[0,:])))
    
    
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

print("Computing features...")    

for img in train_data:
    
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

def test_model(X):
    
    print("Testing model...")    
    
    # Loading in the saved model
    filename = 'logreg.sav'

    clf = pickle.load(open(filename, 'rb'))
    
    preds = clf.predict(X)
    
    print("Prediction vector", "\n", preds)
    
    np.save('predictions.npy', preds)
    
    return preds
    
test_model(X)


