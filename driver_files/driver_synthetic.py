#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 18:58:19 2022

@author: vinco
"""

#%% CLEAR WORKSPACE
from IPython import get_ipython
get_ipython().magic('reset -sf')

#%% IMPORT STANDARD MODULES
import __includes__

import os
import matplotlib.pyplot as plt
import time
from utils.dynsys_utils import calc_dtheta_EVT, embed, calc_lyap_spectrum, \
	calc_dim_Cao1997, _calc_tangent_map
from utils.save_utils import save_pickle
import numpy as np
from scipy.signal import find_peaks
from csaps import csaps

plt.close('all')
tic_begin = time.time()

dirs = {'main' : os.getcwd(),
		'data' : '/Users/vinco/Work/Data/labquakesde/'
		}
			
#%% SELECT CASE STUDY AND SET UP DIRECTORIES
#diff_eq_type = 'param4'
#diff_eq_type = 'param5'
#diff_eq_type = 'param6'
diff_eq_type = 'param7'

#Anoise = 0 # Observational noise amplitude in Pa
Anoise = 4e3 # Observational noise amplitude in Pa

case_studies = ['synthetic/'+diff_eq_type+'/b724', \
				'synthetic/'+diff_eq_type+'/b722', \
				'synthetic/'+diff_eq_type+'/b696', \
				'synthetic/'+diff_eq_type+'/b726', \
				'synthetic/'+diff_eq_type+'/b694', \
				'synthetic/'+diff_eq_type+'/b697', \
				'synthetic/'+diff_eq_type+'/b693', \
				'synthetic/'+diff_eq_type+'/b698', \
				'synthetic/'+diff_eq_type+'/b695', \
				'synthetic/'+diff_eq_type+'/b728', \
				'synthetic/'+diff_eq_type+'/b721', \
				'synthetic/'+diff_eq_type+'/b725', \
				'synthetic/'+diff_eq_type+'/b727', \
				'synthetic/'+diff_eq_type+'/i417']

sigman = [0]*len(case_studies)

#%% SET FLAGS
flag_filter       = False
flag_save         = True
flag_load_data    = True
flag_calc_EVT     = True
flag_calc_LEs     = True

#%% SET PARAMETERS
Ncases = len(case_studies)

v0 = 10e-6
L = 3e-6
Dt = 200

# Embedding dimensions to test
m_list = [3,4,5,6,7,8,9,10,11,12,13,14,15,20,50,100]
Nm = len(m_list)

# Minimum distance between peaks (add 1 for scipy notation), i.e. if you want
# a minimum distance of 3 use 4
min_dist = [2,3,4,5,6,7,8,9,10,11,16,21]
Nmin_dist = len(min_dist)

tau_delay = []
for tt,mm in enumerate(min_dist):
	tau_delay.append(min_dist[tt]-1)
Ntau_delay = len(tau_delay)

if flag_filter==True:
	smoothes = [0.9]*Ncases
else:
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

print("")
D1 = np.zeros((Ncases,Nm,Nmin_dist))
for cc,case_study in enumerate(case_studies):
	
	exp_name = case_study[-5:]
	dirs['pickles'] = dirs['main']+'/../scenarios/'+case_study+'/pickles/'
	dirs['figures'] = dirs['main']+'/../scenarios/'+case_study+'/figures/'
	print(case_study)
	
	#% LOAD DATA
	if flag_load_data==True:
		sol_t   = np.loadtxt(dirs['data']+case_study+'/sol_t.txt')
		sol_u   = np.loadtxt(dirs['data']+case_study+'/sol_u.txt')
		a       = np.loadtxt(dirs['data']+case_study+'/a.txt')
		sigman0 = np.loadtxt(dirs['data']+case_study+'/sigman0.txt')
		v0      = np.loadtxt(dirs['data']+case_study+'/v0.txt')
		L1      = np.loadtxt(dirs['data']+case_study+'/L1.txt')
		tau0    = np.loadtxt(dirs['data']+case_study+'/tau0.txt')
		mu0     = tau0/sigman0
		DT = Dt*v0/L1
		Tend = sol_t[-1]
		Tmin = Tend - DT
		tmin = Tmin*L1/v0
		
		sigman[cc] = sigman0/1e6
		
		#% OBSERVED DATA
		tobs = sol_t[sol_t>Tmin]*L1/v0
		tobs = tobs - tobs[0]
		Nt = tobs.shape[0]
		vobs = v0*np.exp(sol_u[sol_t>Tmin,0])
		ShearStressobs = np.zeros((Nt,))
		ShearStressobs = (tau0 + a*sigman0*sol_u[sol_t>Tmin,1] + \
					Anoise*np.random.randn(Nt))/1e6
		NormStressobs = sigman[cc] + (sigman[cc]*a/mu0)*sol_u[sol_t>Tmin,2]
		
		dtobs = tobs[1]-tobs[0]
		
		p = np.polyfit(tobs, ShearStressobs, deg=1)
		ShearStressobs_det = ShearStressobs - (p[0]*tobs + p[1])
		
		del p
		
		Xobs = np.array([ShearStressobs_det]).T
		Nt,Nx = Xobs.shape
		X = np.zeros((Nt,Nx))
		for nn in np.arange(Nx):
			X[:,nn] = csaps(tobs, Xobs[:,nn], tobs, smooth=smoothes[nn])
		X_norm = np.zeros((Nt,Nx))
		for ii in np.arange(Nx):
			X_norm[:,ii] = (X[:,ii]-np.min(X[:,ii])) / \
				(np.max(X[:,ii])-np.min(X[:,ii]))
	
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
					mhat_tmp, E1, E2 = calc_dim_Cao1997(X=X_norm,
								 tau=[tau_delay], m=np.arange(1,mmax+1,1),
								 E1_thresh=E1_thresh, E2_thresh=E2_thresh,
								 qw=None, flag_single_tau=False,
								 parallel=False)
				
				H, tH = embed(X_norm, tau=[tau_delay],
									 m=[int(mhat_tmp[0][0])], t=tobs)
				LEs_tmp, LEs_mean_tmp, LEs_std_tmp, eps_over_L_tmp = \
					calc_lyap_spectrum(H, sampling=LEs_sampling,
						   eps_over_L0=eps_over_L0, n_neighbors=20)
				mhat.append(mhat_tmp)
				LEs.append(LEs_tmp)
				LEs_mean.append(LEs_mean_tmp)
				LEs_std.append(LEs_std_tmp)
				eps_over_L.append(eps_over_L_tmp)
			else:
				for tautau in np.arange(Ntau_delay):
					mhat_tmp, E1, E2 = calc_dim_Cao1997(X=X_norm,
								tau=[tau_delay[tautau]],
									m=np.arange(1,mmax+1,1),
									E1_thresh=E1_thresh, E2_thresh=E2_thresh,
									qw=None, flag_single_tau=True,
									parallel=False)
					mhat[tautau] = mhat_tmp[0][0]
					if np.isnan(mhat_tmp[0][0]):
						eps_over_L[tautau] = np.nan
					else:
						H, tH = embed(X_norm, tau=[tau_delay[tautau]],
									 m=[int(mhat_tmp[0][0])], t=tobs)
						_, eps_over_L[tautau] = _calc_tangent_map(H,
										   n_neighbors=20, eps_over_L0=eps_over_L0)
				# Define the best embedding parameters as those such that the
				# tangent map for the calculation of the Lyapunov spectrum is
				# calculated using the smallest radius in order to have the
				# required number of neighbros
				tau_best = int(tau_delay[np.nanargmin(eps_over_L)])
				mhat_best = int(mhat[np.nanargmin(eps_over_L)])
				print("m = %d, tau = %d" %(mhat_best, tau_best))
				H, tH = embed(X_norm, tau=[tau_best],
									 m=[mhat_best], t=tobs)
				LEs, LEs_mean, LEs_std, eps_over_L = \
							calc_lyap_spectrum(H, sampling=LEs_sampling,
										eps_over_L0=eps_over_L0, n_neighbors=20)
				
		NtHmax = 0
		for dd,mmin_dist in enumerate(min_dist):
			ind_peaks = find_peaks(X[:,0],distance=mmin_dist)[0]
			
			# EMBEDDING TO ESTIMATE MAXIMUM NtH
			for kk,mm in enumerate(m_list):
				try:
					H, tH = embed(X[ind_peaks,:], tau=[1],
								 m=[mm],t=tobs[ind_peaks])
					NtHmax = np.max([NtHmax,tH.shape[0]])
				except:
					print('mm is too large: skipped')
		
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
									calc_dtheta_EVT(H, p=p_EVT,
													q0_thresh=q0_thresh[kk,dd],
													dq_thresh=dq_thresh_EVT,
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
				save_pickle(file_path=dirs['pickles']+'EVT_ShearStress_det_'+\
					 str('%.2e' %(Anoise))+'.pickle', list_vars=[tH, \
							   d1, theta, qs_thresh, p_EVT, q0_thresh, \
							   min_dist, m_list, case_study, \
							   ShearStressobs_det, smoothes[cc]],\
					 message="Pickling EVT variables: ")
			if flag_calc_LEs==True:
				print("")
				save_pickle(file_path=dirs['pickles']+'LEs_ShearStress_det_'+\
					 str('%.2e' %(Anoise))+'.pickle', \
						 list_vars=[mhat_best, tau_best, LEs, LEs_mean, \
							  LEs_std, eps_over_L],\
					 message="Pickling LEs variables: ")
