#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 13:01:05 2021

@author: Adriano Gualandi
Istituto Nazionale di Geofisica e Vulcanologia
Osservatorio Nazionale Terremoti
via di Vigna Murata 605
00143, Rome, Italy
e-mail: adriano.gualandi@ingv.it
"""

#%% CLEAR WORKSPACE
from IPython import get_ipython
# get_ipython().magic('reset -sf')

#%% IMPORT STANDARD MODULES
import __includes__

import os
import time
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from utils.load_utils import import_data
from utils.param_utils import set_param
from utils.save_utils import save_pickle
from utils.dynsys_utils import \
	calc_dtheta_EVT, \
	embed, calc_lyap_spectrum, \
	calc_dim_Cao1997, \
	_calc_tangent_map

plt.close('all')
tic_begin = time.time()

dirs = {'main' : os.getcwd(),
		'data' : '/Users/vinco/Work/Data/labquakesde/'
		}
			
#%% SELECT CASE STUDY AND SET UP DIRECTORIES
case_studies = ['labquakes/MeleVeeduetal2020/b724',\
				'labquakes/MeleVeeduetal2020/b722',\
				'labquakes/MeleVeeduetal2020/b696',\
				'labquakes/MeleVeeduetal2020/b726',\
				'labquakes/MeleVeeduetal2020/b694',\
				'labquakes/MeleVeeduetal2020/b697',\
				'labquakes/MeleVeeduetal2020/b693',\
				'labquakes/MeleVeeduetal2020/b698',\
				'labquakes/MeleVeeduetal2020/b695',\
				'labquakes/MeleVeeduetal2020/b728',\
				'labquakes/MeleVeeduetal2020/b721',\
				'labquakes/MeleVeeduetal2020/b725',\
				'labquakes/MeleVeeduetal2020/b727',\
				'labquakes/MeleVeeduetal2020/i417']
sigman = [13.6, 14.0, 14.4, 15.0, 15.4, 16.4, 17.0, \
		  17.4, 18.0, 20.0, 22.0, 23.0, 24.0, 25.0]

#%% SET FLAGS
flag_save         = True
flag_load_data    = True
flag_calc_EVT     = True
flag_calc_LEs     = True

#%% SET PARAMETERS
Ncases = len(case_studies)

# Embedding dimensions to test
m_list = [3,4,5,6,7,8,9,10,11,12,13,14,15,20,50,100]
Nm = len(m_list)

# Minimum distance between peaks (add 1 for scipy notation), i.e. if you want
# a minimum distance of 3 use 4
min_dist = [2,3,4,5,6,7,8,9,10,11,16,21]
Nmin_dist = len(min_dist)

tau_delay = []
tt=-1
for mm in min_dist:
	tt+=1
	tau_delay.append(min_dist[tt]-1)
Ntau_delay = len(tau_delay)

smoothes = [1.0]*Ncases

p_EVT         = 2
n_neighbors_min_EVT = 100
dq_thresh_EVT = None
fit_distr     = False

LEs_sampling = ['rand',None]
eps_over_L0 = 0.05

maxtau = 4000
mmax = 20
E1_thresh = 0.9
E2_thresh = 0.9

D1 = np.zeros((Ncases,Nm,Nmin_dist))
for cc,case_study in enumerate(case_studies):
	dirs['pickles'] = dirs['main']+'/../scenarios/'+case_study+'/pickles/'
	dirs['figures'] = dirs['main']+'/../scenarios/'+case_study+'/figures/'
	print("")
	print("")
	print(case_study)
	#% LOAD DATA
	exp_name = case_study[-4:]
	filename = dirs['data']+case_study+'/'+exp_name+'.txt'
	parameters = set_param(case_study)
	
	if flag_load_data==True:
		data = import_data(dirs,filename,parameters,\
					 struct='MeleVeeduetal2020',file_format='txt')
		
		fields = ['Time (s)', 'Loading Point Displacement (microm)', \
			   'Layer Thickness (mm)', 'Shear Stress (MPa)', \
			   'Normal Stress (MPa)', 'Elastic Corrected Displacement (mm)', \
			   'Friction coefficient mu', 'Shear Strain']
		
		#% OBSERVED DATA
		tobs           = data['Time']-data['Time'][0]
		LPDispobs      = data['LPDisp']
		LayerThickobs  = data['LayerThick']
		ShearStressobs = data['ShearStress']
		NormStressobs  = data['NormStress']
		ecDispobs      = data['ecDisp']
		muobs          = data['mu']
		ShearStrainobs = data['ShearStrain']
		
		dtobs = tobs[1]-tobs[0]
		
		p = np.polyfit(tobs, ShearStressobs, deg=1)
		ShearStressobs_det = ShearStressobs - (p[0]*tobs + p[1])
		del p
		
		X = np.array([ShearStressobs_det]).T
		Nt,Nx = X.shape
		X_norm = np.zeros((Nt,Nx))
		for ii in np.arange(Nx):
			X_norm[:,ii] = (X[:,ii]-np.min(X[:,ii])) / (np.max(X[:,ii])-np.min(X[:,ii]))
		
		# LYAPUNOV SPECTRUM
		if flag_calc_LEs==True:
			mhat = []
			LEs        = []
			LEs_mean   = []
			LEs_std    = []
			eps_over_L = np.zeros((Ntau_delay,1))
			mhat       = np.zeros((Ntau_delay,1))
			
			# BEST EMBEDDING PARAMETERS
			# m FROM CAO (1997) [tau as minimum tau for which m is determined]
			if tau_delay==0:
				mhat_tmp = [[np.nan]]
				while np.isnan(mhat_tmp[0][0]):
					tau_delay+=1
					mhat_tmp, E1, E2 = calc_dim_Cao1997(X=X_norm, tau=[tau_delay], \
			 					m=np.arange(1,mmax+1,1),E1_thresh=E1_thresh, \
			 					E2_thresh=E2_thresh, qw=None, \
			 					flag_single_tau=False, parallel=False)
				
				H, tH = embed(X_norm, tau=[tau_delay], \
									 m=[int(mhat_tmp[0][0])], t=tobs)
				LEs_tmp, LEs_mean_tmp, LEs_std_tmp, eps_over_L_tmp = \
					calc_lyap_spectrum(H, sampling=LEs_sampling, \
						   eps_over_L0=eps_over_L0, n_neighbors=20)
				mhat.append(mhat_tmp)
				LEs.append(LEs_tmp)
				LEs_mean.append(LEs_mean_tmp)
				LEs_std.append(LEs_std_tmp)
				eps_over_L.append(eps_over_L_tmp)
			else:
				for tautau in np.arange(Ntau_delay):
					mhat_tmp, E1, E2 = calc_dim_Cao1997(X=X_norm, \
								tau=[tau_delay[tautau]], m=np.arange(1,mmax+1,1), \
								E1_thresh=E1_thresh, E2_thresh=E2_thresh, \
								qw=None, flag_single_tau=True, parallel=False)
					mhat[tautau] = mhat_tmp[0][0]
					if np.isnan(mhat_tmp[0][0]):
						eps_over_L[tautau] = np.nan
					else:
						H, tH = embed(X_norm, tau=[tau_delay[tautau]], \
									 m=[int(mhat_tmp[0][0])], t=tobs)
						_, eps_over_L[tautau] = _calc_tangent_map(H, \
										   n_neighbors=20, eps_over_L0=eps_over_L0)
				# Define the best embedding parameters as those such that the
				# tangent map for the calculation of the Lyapunov spectrum is
				# calculated using the smallest radius in order to have the
				# required number of neighbors
				tau_best = int(tau_delay[np.nanargmin(eps_over_L)])
				mhat_best = int(mhat[np.nanargmin(eps_over_L)])
				print("m = %d, tau = %d" %(mhat_best, tau_best))
				H, tH = embed(X_norm, tau=[tau_best], \
									 m=[mhat_best], t=tobs)
				LEs, LEs_mean, LEs_std, eps_over_L = \
							calc_lyap_spectrum(H, sampling=LEs_sampling, \
										eps_over_L0=eps_over_L0, n_neighbors=20)
				
		NtHmax = 0
		for dd,mmin_dist in enumerate(min_dist):
			ind_peaks = find_peaks(X[:,0],distance=mmin_dist)[0]
			
			# EMBEDDING TO ESTIMATE MAXIMUM NtH
			kk = -1
			for mm in m_list:
				kk+=1
				H, tH = embed(X[ind_peaks,:], tau=[1], \
							 m=[mm],t=tobs[ind_peaks])
				NtHmax = np.max([NtHmax,tH.shape[0]])
		
		tH        = np.zeros((Nm,Nmin_dist,NtHmax))*np.nan
		d1        = np.zeros((Nm,Nmin_dist,NtHmax))*np.nan
		theta     = np.zeros((Nm,Nmin_dist,NtHmax))*np.nan
		qs_thresh = np.zeros((Nm,Nmin_dist,NtHmax))*np.nan
		q0_thresh = np.zeros((Nm,Nmin_dist))*np.nan
		for dd,mmin_dist in enumerate(min_dist):
			ind_peaks = find_peaks(X[:,0],distance=mmin_dist)[0]
			
			
			# EVT CALCULATION
			for kk,mm in enumerate(m_list):
				try:
					H, tH_tmp = embed(X[ind_peaks,:], tau=[1], \
								 m=[mm],t=tobs[ind_peaks])
					NtH = tH_tmp.shape[0]
					if flag_calc_EVT==True:
						q0_thresh[kk,dd] = 1 - n_neighbors_min_EVT/NtH
						d1_tmp,theta_tmp,qs_thresh_tmp = \
									calc_dtheta_EVT(H, p=p_EVT, \
							q0_thresh=q0_thresh[kk,dd], dq_thresh=dq_thresh_EVT,\
							fit_distr=fit_distr)
						
						tH[kk,dd,:NtH]        = tH_tmp
						d1[kk,dd,:NtH]        = d1_tmp[:,0]
						theta[kk,dd,:NtH]     = theta_tmp[:,0]
						qs_thresh[kk,dd,:NtH] = qs_thresh_tmp[:,0]
						
						D1[cc,kk,dd] = np.nanmean(d1[kk,dd,:NtH])
				except:
					tH[kk,dd,:NtH]        = np.nan
					d1[kk,dd,:NtH]        = np.nan
					theta[kk,dd,:NtH]     = np.nan
					qs_thresh[kk,dd,:NtH] = np.nan
					D1[cc,kk,dd] = np.nan
		
		if flag_save==True:
			if flag_calc_EVT==True:
				print("")
				save_pickle(file_path=dirs['pickles']+'EVT_ShearStress_det'+\
					 '.pickle', list_vars=[tH, \
							   d1, theta, qs_thresh, p_EVT, q0_thresh, \
							   min_dist, m_list, case_study, \
							   ShearStressobs_det, smoothes[cc]],\
					 message="Pickling EVT variables: ")
			if flag_calc_LEs==True:
				print("")
				save_pickle(file_path=dirs['pickles']+'LEs_ShearStress_det'+\
					 '.pickle', list_vars=[mhat_best, tau_best, LEs, LEs_mean, \
							   LEs_std, eps_over_L],\
					 message="Pickling LEs variables: ")
					
