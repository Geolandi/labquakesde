#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 09:52:32 2021

@author: adriano
"""
import pickle
import time

def save_pickle(file_path,list_vars,message='Pickling variable: '):
	# file_path: String indicating the absolute path where to save the
	#            variables
	# list_vars: List containing the variables to be pickled
	tic = time.time()
	print(message, end='')
	with open(file_path, 'wb') as f:
		pickle.dump(list_vars, f)
	print("%.2f s" %(time.time()-tic))