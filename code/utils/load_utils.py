#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 12:42:40 2021

@author: vinco
"""

import numpy as np
import pickle
import time

def import_data(dirs,filename,parameters,struct=None,file_format=None):
	print("")
	print("Loading variables and setting parameters: ", end='')
	time.sleep(0.5)
	if struct=='MeleVeeduetal2020':
		if file_format=='txt':
			f = open(filename, 'r')
			L=0
			for line in f:
				L+=1
			
			f.close()
			
			Nheaders = parameters['Nheaders']
			L=L-Nheaders
			Rec           = np.empty([L,1])
			LPDisp        = np.empty([L,1])
			LayerThick    = np.empty([L,1])
			ShearStress   = np.empty([L,1])
			NormStress    = np.empty([L,1])
			OnBoard       = np.empty([L,1])
			Time          = np.empty([L,1])
			Rec_float     = np.empty([L,1])
			TimeOnBoard   = np.empty([L,1])
			ecDisp        = np.empty([L,1])
			mu            = np.empty([L,1])
			ShearStrain   = np.empty([L,1])
			slip_velocity = np.empty([L,1])
			ll=-1
			tt=-1
			f = open(filename, 'r')
			for line in f:
				ll+=1
				columns = line.split()
				if ll>Nheaders-1:
					tt+=1
					Rec[tt]           = int(columns[0])
					LPDisp[tt]        = float(columns[1])
					LayerThick[tt]    = float(columns[2])
					ShearStress[tt]   = float(columns[3])
					NormStress[tt]    = float(columns[4])
					OnBoard[tt]       = float(columns[5])
					Time[tt]          = float(columns[6])
					Rec_float[tt]     = float(columns[7])
					TimeOnBoard[tt]   = float(columns[8])
					ecDisp[tt]        = float(columns[9])
					mu[tt]            = float(columns[10])
					ShearStrain[tt]   = float(columns[11])
					slip_velocity[tt] = float(columns[12])
				
			f.close()
			
			ind_keep = [i for i,x in enumerate(Time[:,0]) if x>=parameters['t0'] and x<=parameters['tend']]
			
			data_output = {
					'Rec' : Rec[ind_keep,0],
					'LPDisp' : LPDisp[ind_keep,0],
					'LayerThick' : LayerThick[ind_keep,0],
					'ShearStress' : ShearStress[ind_keep,0],
					'NormStress' : NormStress[ind_keep,0],
					'OnBoard' : OnBoard[ind_keep,0],
					'Time' : Time[ind_keep,0],
					'Rec_float' : Rec_float[ind_keep,0],
					'TimeOnBoard' : TimeOnBoard[ind_keep,0],
					'ecDisp' : ecDisp[ind_keep,0],
					'mu' : mu[ind_keep,0],
					'ShearStrain' : ShearStrain[ind_keep,0],
					'OnBoarddot' : slip_velocity[ind_keep,0],
					}
	print("DONE")
	return data_output
		


def load_pickle(file_path, message='Loading pickled variables: '):
	# file_path: String indicating the absolute path from where to load the
	#            variables
	tic = time.time()
	if file_path[-7:]!='.pickle':
		file_path = file_path+'.pickle'
	print("")
	print(message, end='')
	with open(file_path, 'rb') as f:
		list_vars = pickle.load(f)
	return list_vars
	print("%.2f s" %(time.time()-tic))