'''
Dilated convolution for the multidimensional delay vectors/ v1.1. split
'''

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import argparse

'''
Parameters Definition
dim: time series dimension
delay_s: time series segment length
n_class: number of labels
batch_size: batch size
num_epoch: number of epochs
lr1: learning rate 
l2: regularization
'''
dim=3
delay_s=100
n_class=6
batch_size=256
num_epoch=600
lr1=0.0001
l2=0.00001
'''##########################'''

def optimistic_restore(session, save_file):		
	ckpt_o=tf.train.get_checkpoint_state(save_file)
	reader = tf.train.NewCheckpointReader(ckpt_o.model_checkpoint_path)
	saved_shapes = reader.get_variable_to_shape_map()
	var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
			if var.name.split(':')[0] in saved_shapes])
	restore_vars = []
	name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
	with tf.variable_scope('', reuse=True):
		for var_name, saved_var_name in var_names:
			curr_var = name2var[saved_var_name]
			var_shape = curr_var.get_shape().as_list()
			if var_shape == saved_shapes[saved_var_name]:
				restore_vars.append(curr_var)
	saver = tf.train.Saver(restore_vars)
	saver.restore(session, ckpt_o.model_checkpoint_path)

def f_score(recall,precision):
	f=2*(recall*precision)/(recall+precision)
	return f


#TP+FN
def tp_fn(label_v):
	a=[]
	for i in range(n_class):
		ai=np.where(label_v[:,i]==1)
		a.append(ai)		
	a_size=[]
	for i in range(n_class):
		ai_size=a[i][0].shape[0]
		a_size.append(ai_size)
	return a_size


#TP+FP
def tp_fp(pred_v):
	a_t=[]
	for i in range(n_class):
		ai_t=np.where((np.argmax(pred_v,1)==i))
		a_t.append(ai_t)		
	a_p=[]
	for i in range(n_class):
		ai_p=a_t[i][0].shape[0]		
		a_p.append(ai_p)
	return a_p

	
def batch_data(data,label,batch_size):
	assert data.shape[0]==label.shape[0]
	indices=np.arange(data.shape[0])
	np.random.shuffle(indices)
	for start_idx in range(0,data.shape[0]-batch_size+1,batch_size):
		excerpt=indices[start_idx:start_idx+batch_size]
		yield data[excerpt],label[excerpt]

def conv2d(inp,weight,bias,dilation,padding,stride,name):
	with tf.variable_scope(name):
		regularizer = tf.contrib.layers.l2_regularizer(scale=l2)
		out1=tf.layers.conv2d(inp,weight[-1],(weight[0],weight[1]),strides=(1,stride),padding=padding,dilation_rate=(dilation[1],dilation[2]),activation=tf.nn.leaky_relu,kernel_regularizer=regularizer)
		return out1
		
def dense_layer(input1, shape_w,shape_b,name):
	with tf.variable_scope(name):
		w =tf.get_variable('w'+name,shape_w,initializer=tf.contrib.layers.xavier_initializer()) 
		b=tf.get_variable('b'+name,shape_b,initializer=tf.constant_initializer(0.1))
		pred_activation = tf.nn.xw_plus_b(input1,w,b)
		return pred_activation

PATH='./datasets'
MDPATH='./WISDM_v1.1/split/models'
RESULT_PATH='./WISDM_v1.1/split/results.csv'

feat_train=np.load(PATH+'/TRAIN_feat_sp_v1_100.npy')
label_train=np.load(PATH+'/TRAIN_label_sp_v1_100.npy')

feat_test=np.load(PATH+'/TEST_feat_sp_v1_100.npy')
label_test=np.load(PATH+'/TEST_label_sp_v1_100.npy')
print("features - labels shape:",feat_train.shape,label_train.shape)

feat_train=np.expand_dims(feat_train,axis=3)
feat_test=np.expand_dims(feat_test,axis=3)

if not os.path.exists(MDPATH):
	os.makedirs(MDPATH)

inp=tf.placeholder(tf.float32,[batch_size,dim,delay_s,1])
target=tf.placeholder(tf.int32,[batch_size,n_class])
lr=tf.placeholder(tf.float32,())

out=conv2d(inp,weight=[3,10,1,32],bias=[32],padding='SAME',stride=1,dilation=[1,1,2,1],name='dil_1')
print("layer1_dil:\t",out.get_shape())

out=conv2d(out,weight=[1,4,32,32],bias=[32],padding='VALID',stride=4,dilation=[1,1,1,1],name='pool_2')
print("layer2_pool:\t",out.get_shape())

out=conv2d(out,weight=[3,3,32,32],bias=[32],padding='SAME',stride=1,dilation=[1,1,2,1],name='dil_3')
print("layer3_dil:\t",out.get_shape())

out=conv2d(out,weight=[1,2,32,32],bias=[32],padding='VALID',stride=2,dilation=[1,1,1,1],name='pool_4')
print("layer4_pool:\t",out.get_shape())

out_size=out.get_shape().as_list()
out_flat=tf.reshape(out,(out_size[0],-1))

out_flat=dense_layer(out_flat,shape_w=[out_size[1]*out_size[2]*out_size[3],1024],shape_b=[1024],name='dense')
out_flat=tf.nn.relu(out_flat)

out_final=dense_layer(out_flat,shape_w=[1024,n_class],shape_b=[n_class],name='last')
print("out final: ",out_final.get_shape())

pred=tf.nn.softmax(out_final)
err1=tf.nn.softmax_cross_entropy_with_logits_v2(labels=target,logits=out_final)
err=tf.reduce_mean(err1)+tf.losses.get_regularization_loss()
err_val=tf.reduce_mean(err1)
print("error: ",err1.get_shape(),err.get_shape())

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(target,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

global_step=tf.train.get_or_create_global_step()
train_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(err,global_step=global_step)

saver = tf.train.Saver(max_to_keep=10)
init=tf.global_variables_initializer()

with tf.Session() as sess:
	ckpt=tf.train.get_checkpoint_state(MDPATH)
	if ckpt and ckpt.model_checkpoint_path:
		print(ckpt,ckpt.model_checkpoint_path)
		sess.run(init)
		optimistic_restore(sess,MDPATH)
	else:
		sess.run(init)
	epoch=0
	while epoch < num_epoch:
		epoch+=1
		loss_tr=0
		acc_tr=0
		num_batch_tr=0
		for feat_tr,label_tr in batch_data(feat_train,label_train,batch_size):
			num_batch_tr+=1
			args={inp:feat_tr,target:label_tr,lr:lr1}
			_,loss,acc,step1,fin=sess.run([train_op,err,accuracy,global_step,out_final],feed_dict=args)
			loss_tr+=loss
			acc_tr+=acc
			lr1=0.9995*lr1
		
		loss_f_tr=loss_tr/num_batch_tr
		acc_f_tr=acc_tr/num_batch_tr
		a_fake=[0]*n_class
		a_act=[0]*n_class
		a_pred=[1e-8]*n_class
		a_pred_fake=[0]*n_class
		loss_v=0
		acc_v=0
		num_bv=0

		for feat_v,label_v in batch_data(feat_test,label_test,batch_size):
			num_bv+=1
			args={inp:feat_v,target:label_v}
			loss,acc,pred_v=sess.run([err_val,accuracy,out_final],feed_dict=args)
			loss_v+=loss
			acc_v+=acc
			
			a=tp_fn(label_v)	#TP+FN
			for i in range(n_class):
				a_act[i]+=a[i]

			aa=tp_fp(pred_v)	#TP+FP
			for i in range(n_class):
				a_pred[i]+=aa[i]
		
			ll=np.equal(np.argmax(pred_v,1),np.argmax(label_v,1))
			ind=np.where(ll==False)

			for i in ind[0]:	#FN
				for j in range(n_class):
					if label_v[i,j]==1:
						a_fake[j]+=1
			for i in ind[0]:	#FP
				for j in range(n_class):
					if np.argmax(pred_v[i])==j:
						a_pred_fake[j]+=1

		loss_fv=loss_v/num_bv
		acc_fv=acc_v/num_bv
		a_fake=np.asarray(a_fake)
		a_act=np.asarray(a_act)
		a_pred=np.asarray(a_pred)
		a_pred_fake=np.asarray(a_pred_fake)
		a_rcal=1-a_fake/a_act
		a_prc=1-a_pred_fake/a_pred
		a_fs=f_score(a_rcal,a_prc)
		
		print("epoch:",epoch,":   loss_tr",loss_f_tr,"   acc_tr:",acc_f_tr,"   loss_val:",loss_fv,"   acc_val:",acc_fv,"   f_score:",a_fs)
		with open(RESULT_PATH, 'a') as f2:
			f2.write(str(epoch) + '  ' + str(loss_f_tr)  + '  ' + str(acc_f_tr) + '  ' + str(loss_fv) + '  ' + str(acc_fv)+ '  ' + str(a_fs)  +'\n')
		saver.save(sess, save_path=MDPATH + '/model_'+str(step1)+'.ckpt',global_step=step1,write_state=True)
	
		








