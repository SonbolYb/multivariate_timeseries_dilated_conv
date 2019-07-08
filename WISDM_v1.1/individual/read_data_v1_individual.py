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
	mean_value = df[feature_name].values.mean()
	print(max_value,min_value)
	df[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
 

def create_delay(data_tr,path_dest,name):

	
	SEGMENT=200
	STEP=20
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
		label1 =stats.mode(data_tr['activity'].values[i: i + SEGMENT])[0][0] 
		labels.append(label1)

	# Convert to numpy
	data_convoluted = np.asarray(data_convoluted, dtype=np.float32)
	
	# One-hot encoding
	label_dummy=indices_to_one_hot(labels,n_class)
	labels = np.asarray(label_dummy, dtype=np.float32)
	print("X "+name+" size: ", data_convoluted.shape)
	print("y "+name+" size:", labels.shape)

	np.save(path_dest+'/'+name+'_feat_in_v1_200',data_convoluted)
	np.save(path_dest+'/'+name+'_label_in_v1_200',labels)

PATH='./datasets/WISDM_at_v1.1_raw.txt'
PATH_DEST='./datasets'

data=pd.read_csv(PATH,names=['id','activity','timestamp','x','y','z'], header=None)#,error_bad_lines=False)
data['z'].replace({';': ''}, regex=True, inplace=True)
data['z']=data['z'].astype(float)
data = data.dropna()

data = data[data.x < 20]
data =data[data.x > -20]

data = data[data.y < 20]
data = data[data.y > -20]

data = data[data.z < 20]
data = data[data.z > -20]

data['activity']=data['activity'].map({'Walking':0,'Jogging':1,'Upstairs':2,'Downstairs':3,'Sitting':4,'Standing':5})

set_id=list(set(data['id']))
len_id=len(set_id)

#divide set
id_train=set_id[:28]

test=data.loc[~data['id'].isin(id_train)].reset_index()
train=data.loc[data['id'].isin(id_train)].reset_index()

create_delay(train,PATH_DEST,'TRAIN')
create_delay(test,PATH_DEST,'TEST')

