'''
train and test sets
'''

import numpy as np
import pandas as pd
import csv
from scipy import stats
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

n_class=6

def indices_to_one_hot(data, nb_classes):
	targets = np.array(data).reshape(-1)
	out=np.eye(nb_classes)[targets]
	return out

def normalize(df,feature_name):
	max_value = df[feature_name].values.max()
	min_value = df[feature_name].values.min()
	df[feature_name] = 2*(df[feature_name] - min_value) / (max_value - min_value)-1
 

def create_delay(data_tr,path_dest):

	SEGMENT=200
	STEP=200
	data_convoluted = []
	labels = []
	for i in range(0,len(data_tr)-SEGMENT,STEP):
		
		ids=list(data_tr['id'].values[i:i+SEGMENT])
		if len(set(ids))>1:		#to find different subject; a delay vector is from one subject
			pass

		x=data_tr['x'].values[i:i+SEGMENT]
		y=data_tr['y'].values[i:i+SEGMENT]
		z=data_tr['z'].values[i:i+SEGMENT]
		data_convoluted.append([x, y, z])
		
		# Label for a data window is the label that appears most commonly
		label =	stats.mode(data_tr['activity'].values[i: i + SEGMENT])[0][0] 
		labels.append(label)

	# Convert to numpy
	data_convoluted = np.asarray(data_convoluted, dtype=np.float32)
	
	# One-hot encoding
	label_dummy=indices_to_one_hot(labels,n_class)
	labels = np.asarray(label_dummy, dtype=np.float32)
	
	x_train, x_test, y_train, y_test = train_test_split(data_convoluted, labels, test_size=0.3, random_state=13)
	print("X TRAIN size: ", len(x_train))
	print("y TRAIN size: ", len(y_train))
	print("X TEST size: ", len(x_test))
	print("y TEST size: ", len(y_test))
	
	np.save(path_dest+'/TRAIN_feat_sp_v2_200',x_train)
	np.save(path_dest+'/TRAIN_label_sp_v2_200',y_train)
	
	np.save(path_dest+'/TEST_feat_sp_v2_200',x_test)
	np.save(path_dest+'/TEST_label_sp_v2_200',y_test)

PATH='./datasets/WISDM_at_v2.0_raw.txt'
PATH_DEST='./datasets'

data=pd.read_csv(PATH,names=['id','activity','timestamp','x','y','z'], header=None)
data['z'].replace({';': ''}, regex=True, inplace=True)
data['z']=data['z'].astype(float)
data = data.dropna()

data = data[data.x < 20]
data =data[data.x > -20]

data = data[data.y < 20]
data = data[data.y > -20]

data = data[data.z < 20]
data = data[data.z > -20]

data['activity']=data['activity'].map({'Walking':0,'Jogging':1,'Sitting':2,'Standing':3,'Stairs':4,'LyingDown':5})

create_delay(data,PATH_DEST)


