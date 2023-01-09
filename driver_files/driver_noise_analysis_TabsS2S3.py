#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 14:28:38 2022

@author: vinco
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
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from utils.load_utils import import_data
from utils.param_utils import set_param
from csaps import csaps

plt.close('all')
tic_begin = time.time()

dirs = {'main' : os.getcwd(),
		'data' : '/Users/vinco/Work/Data/labquakesde/'
		}
			
#%% SELECT CASE STUDY AND SET UP DIRECTORIES
case_studies = ['labquakes/MeleVeeduetal2020/b724', \
				'labquakes/MeleVeeduetal2020/b722', \
				'labquakes/MeleVeeduetal2020/b696', \
				'labquakes/MeleVeeduetal2020/b726', \
				'labquakes/MeleVeeduetal2020/b694', \
				'labquakes/MeleVeeduetal2020/b697', \
				'labquakes/MeleVeeduetal2020/b693', \
				'labquakes/MeleVeeduetal2020/b698', \
				'labquakes/MeleVeeduetal2020/b695', \
				'labquakes/MeleVeeduetal2020/b728', \
				'labquakes/MeleVeeduetal2020/b721', \
				'labquakes/MeleVeeduetal2020/b725', \
				'labquakes/MeleVeeduetal2020/b727', \
				'labquakes/MeleVeeduetal2020/i417']
sigman = [13.6, 14.0, 14.4, 15.0, 15.4, 16.4, 17.0, \
		  17.4, 18.0, 20.0, 22.0, 23.0, 24.0, 25.0]

#%% SET FLAGS
flag_load_data    = True

#%% SET PARAMETERS
cc = -1
Ncases = len(case_studies)
smoothes = [0.9]*Ncases

min_dist = 10
Twin = 200

W_sigman = np.zeros((Ncases,))
pvalue_sigman = np.zeros((Ncases,))
W_tauf = np.zeros((Ncases,))
pvalue_tauf = np.zeros((Ncases,))
sigman0 = np.zeros((Ncases,))
epsilon_NormStress = np.zeros((Ncases,))
epsilon_ShearStress = np.zeros((Ncases,))
epsilon_LayerThick = np.zeros((Ncases,))
epsilon_mu = np.zeros((Ncases,))
epsilon_ecDisp = np.zeros((Ncases,))
epsilon_ShearStrain = np.zeros((Ncases,))
dmax_ShearStressobs = np.zeros((Ncases,))
dmax_LayerThickobs = np.zeros((Ncases,))
dmax_muobs = np.zeros((Ncases,))
dmax_ecDispobs = np.zeros((Ncases,))
dmax_ShearStrainobs = np.zeros((Ncases,))
ratio_epsilon_dmax_ShearStressobs = np.zeros((Ncases,))
ratio_epsilon_dmax_LayerThickobs = np.zeros((Ncases,))
ratio_epsilon_dmax_muobs = np.zeros((Ncases,))
ratio_epsilon_dmax_ecDispobs = np.zeros((Ncases,))
ratio_epsilon_dmax_ShearStrainobs = np.zeros((Ncases,))

std_tauf_inter = np.zeros((Ncases,))
std_sigman_inter = np.zeros((Ncases,))

exp_names = []
tstart = np.zeros((Ncases,))
tend = np.zeros((Ncases,))
sigman0 = np.zeros((Ncases,))
std_sigman0 = np.zeros((Ncases,))
v0 = np.zeros((Ncases,))
std_v0 = np.zeros((Ncases,))
v0_inter = np.zeros((Ncases,))
std_v0_inter = np.zeros((Ncases,))
std_NormStress_inter_norm_smooth = np.zeros((Ncases,))
std_ShearStress_inter_norm_smooth = np.zeros((Ncases,))
std_LayerThick_inter_norm_smooth = np.zeros((Ncases,))
std_ecDisp_inter_norm_smooth = np.zeros((Ncases,))
std_mu_inter_norm_smooth = np.zeros((Ncases,))
std_ShearStrain_inter_norm_smooth = np.zeros((Ncases,))
for case_study in case_studies:
	dirs['pickles'] = dirs['main']+'/../scenarios/'+case_study+'/pickles/'
	dirs['figures'] = dirs['main']+'/../scenarios/'+case_study+'/figures/'
	print("")
	print("")
	print(case_study)
	cc+=1
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
		
		tstart[cc] = data['Time'][0]
		tend[cc] = data['Time'][-1]
		
		dtobs = tobs[1]-tobs[0]
		
		# detrend data
		p = np.polyfit(tobs, LayerThickobs, deg=1)
		LayerThickobs_det = LayerThickobs - (p[0]*tobs + p[1])
		
		p = np.polyfit(tobs, ecDispobs, deg=1)
		ecDispobs_det = ecDispobs - (p[0]*tobs + p[1])
		
		p = np.polyfit(tobs, ShearStressobs, deg=1)
		ShearStressobs_det = ShearStressobs - (p[0]*tobs + p[1])
		
		p = np.polyfit(tobs, muobs, deg=1)
		muobs_det = muobs - (p[0]*tobs + p[1])
		
		p = np.polyfit(tobs, ShearStrainobs, deg=3)
		ShearStrainobs_det = ShearStrainobs - (p[0]*tobs**3 + p[1]*tobs**2 + p[2]*tobs + p[3])
		
		del p
		
		if exp_name == 'i417':
			p, C_p = np.polyfit(tobs, ecDispobs*1e-3, deg=1, cov=True)
		else:
			p, C_p = np.polyfit(tobs, ecDispobs*1e3, deg=1, cov=True)
		v0[cc] = p[0]
		std_v0[cc] = np.sqrt(C_p[0,0])
		
		sigman0[cc] = np.mean(NormStressobs)
		std_sigman0[cc] = np.std(NormStressobs-sigman0[cc])
		
		# smooth and normalize shear stress
		Xobs = np.array([ShearStressobs_det]).T
		Nt,Nx = Xobs.shape
		X = np.zeros((Nt,Nx))
		for nn in np.arange(Nx):
			X[:,nn] = csaps(tobs, Xobs[:,nn], tobs, smooth=smoothes[nn])
		X_norm = np.zeros((Nt,Nx))
		for ii in np.arange(Nx):
			X_norm[:,ii] = (X[:,ii]-np.min(X[:,ii])) / (np.max(X[:,ii])-np.min(X[:,ii]))
		

		NormStressobs_norm = (NormStressobs - \
 		   np.min(NormStressobs)) / \
 			(np.max(NormStressobs)-np.min(NormStressobs))
			 
		ShearStressobs_det_norm = (ShearStressobs_det - \
 		   np.min(ShearStressobs_det)) / \
 			(np.max(ShearStressobs_det)-np.min(ShearStressobs_det))
		LayerThickobs_det_norm = (LayerThickobs_det - \
 		   np.min(LayerThickobs_det)) / \
 			(np.max(LayerThickobs_det)-np.min(LayerThickobs_det))
		muobs_det_norm = (muobs_det - \
 		   np.min(muobs_det)) / \
 			(np.max(muobs_det)-np.min(muobs_det))
		ecDispobs_det_norm = (ecDispobs_det - \
 		   np.min(ecDispobs_det)) / \
 			(np.max(ecDispobs_det)-np.min(ecDispobs_det))
		ShearStrainobs_det_norm = (ShearStrainobs_det - \
 		   np.min(ShearStrainobs_det)) / \
 			(np.max(ShearStrainobs_det)-np.min(ShearStrainobs_det))
		ShearStress_norm_smooth = csaps(tobs, ShearStressobs_det_norm, tobs, smooth=smoothes[cc])
		LayerThick_norm_smooth = csaps(tobs, LayerThickobs_det_norm, tobs, smooth=smoothes[cc])
		ecDisp_norm_smooth = csaps(tobs, ecDispobs_det_norm, tobs, smooth=smoothes[cc])
		mu_norm_smooth = csaps(tobs, muobs_det_norm, tobs, smooth=smoothes[cc])
		ShearStrain_norm_smooth = csaps(tobs, ShearStrainobs_det_norm, tobs, smooth=smoothes[cc])
		
		
		ind_peaks = find_peaks(X[:,0],distance=min_dist)[0]
		
		ind_peaks_neg = find_peaks(-X[:,0],distance=min_dist)[0]
		if ind_peaks[0]>ind_peaks_neg[0]:
			if ind_peaks.shape[0]==ind_peaks_neg.shape[0]:
				ind_peaks = ind_peaks[:-1]
				ind_peaks_neg = ind_peaks_neg[1:]
			elif ind_peaks.shape[0]<ind_peaks_neg.shape[0]:
				ind_peaks_neg = ind_peaks_neg[1:]
			elif ind_peaks.shape[0]>ind_peaks_neg.shape[0]:
				raise ValueError('At least one negative peak missed')
		elif ind_peaks[0]<ind_peaks_neg[0]:
			if ind_peaks.shape[0]<ind_peaks_neg.shape[0]:
				raise ValueError('At least one positive peak missed')
			elif ind_peaks.shape[0]>ind_peaks_neg.shape[0]:
				ind_peaks = ind_peaks[:-1]
		
		for pp in np.arange(ind_peaks.shape[0]):
			twin = np.min([Twin,ind_peaks[pp]])
			ind_peaks[pp] = ind_peaks[pp]-twin+np.argmax(Xobs[ind_peaks[pp]-twin:ind_peaks[pp]+twin,0])
			twin_neg = np.min([Twin,ind_peaks_neg[pp]])
			ind_peaks_neg[pp] = ind_peaks_neg[pp]-twin_neg + \
				np.argmax(-Xobs[ind_peaks_neg[pp]-twin_neg:ind_peaks_neg[pp]+twin_neg,0])
		
		ind_peaks = np.unique(ind_peaks)
		ind_peaks_neg = np.unique(ind_peaks_neg)
		
		Npeaks = ind_peaks.shape[0]
		ind_peaks_center = np.zeros((Npeaks,))
		ind_peaks_center = ind_peaks_center.astype(int)
		for pp in np.arange(Npeaks):
			ind_peaks_center[pp] = int(ind_peaks[pp] + \
				  np.argmin(np.abs(Xobs[ind_peaks[pp]:ind_peaks_neg[pp]+1,0])))
		
		tobs_inter = np.array([])
		ShearStressobs_inter_smooth_err = np.array([])
		NormStressobs_inter_smooth_err = np.array([])
		ShearStressobs_inter_norm_smooth_err = np.array([])
		NormStressobs_inter_norm_smooth_err = np.array([])
		LayerThickobs_inter_norm_smooth_err = np.array([])
		ecDispobs_inter_norm_smooth_err = np.array([])
		muobs_inter_norm_smooth_err = np.array([])
		ShearStrainobs_inter_norm_smooth_err = np.array([])
		ecDisp_inter = np.array([])
		tt = 10
		for pp in np.arange(len(ind_peaks_center)-1):
			tobs_inter_tmp = tobs[ind_peaks_neg[pp]+tt:ind_peaks[pp+1]-tt]
			tobs_inter = np.concatenate((tobs_inter,tobs_inter_tmp))
			
			ShearStressobs_inter = ShearStressobs_det[ind_peaks_neg[pp]+tt:ind_peaks[pp+1]-tt]
			ShearStressobs_inter_smooth = csaps(tobs_inter_tmp, ShearStressobs_inter, tobs_inter_tmp, smooth=smoothes[cc])
			ShearStressobs_inter_smooth_err = np.concatenate((ShearStressobs_inter_smooth_err,ShearStressobs_inter-ShearStressobs_inter_smooth))
			
			NormStressobs_inter = NormStressobs[ind_peaks_neg[pp]+tt:ind_peaks[pp+1]-tt]
			NormStressobs_inter_smooth = csaps(tobs_inter_tmp, NormStressobs_inter, tobs_inter_tmp, smooth=smoothes[cc])
			NormStressobs_inter_smooth_err = np.concatenate((NormStressobs_inter_smooth_err,NormStressobs_inter-NormStressobs_inter_smooth))
			
			ShearStressobs_inter_norm = ShearStressobs_det_norm[ind_peaks_neg[pp]+tt:ind_peaks[pp+1]-tt]
			ShearStressobs_inter_norm_smooth = csaps(tobs_inter_tmp, ShearStressobs_inter_norm, tobs_inter_tmp, smooth=smoothes[cc])
			ShearStressobs_inter_norm_smooth_err = np.concatenate((ShearStressobs_inter_norm_smooth_err,ShearStressobs_inter_norm-ShearStressobs_inter_norm_smooth))
			
			NormStressobs_inter_norm = NormStressobs_norm[ind_peaks_neg[pp]+tt:ind_peaks[pp+1]-tt]
			NormStressobs_inter_norm_smooth = csaps(tobs_inter_tmp, NormStressobs_inter_norm, tobs_inter_tmp, smooth=smoothes[cc])
			NormStressobs_inter_norm_smooth_err = np.concatenate((NormStressobs_inter_norm_smooth_err,NormStressobs_inter_norm-NormStressobs_inter_norm_smooth))
			
			LayerThickobs_inter_norm = LayerThickobs_det_norm[ind_peaks_neg[pp]+tt:ind_peaks[pp+1]-tt]
			LayerThickobs_inter_norm_smooth = csaps(tobs_inter_tmp, LayerThickobs_inter_norm, tobs_inter_tmp, smooth=smoothes[cc])
			LayerThickobs_inter_norm_smooth_err = np.concatenate((LayerThickobs_inter_norm_smooth_err,LayerThickobs_inter_norm-LayerThickobs_inter_norm_smooth))
			
			ecDispobs_inter_norm = ecDispobs_det_norm[ind_peaks_neg[pp]+tt:ind_peaks[pp+1]-tt]
			ecDispobs_inter_norm_smooth = csaps(tobs_inter_tmp, ecDispobs_inter_norm, tobs_inter_tmp, smooth=smoothes[cc])
			ecDispobs_inter_norm_smooth_err = np.concatenate((ecDispobs_inter_norm_smooth_err,ecDispobs_inter_norm-ecDispobs_inter_norm_smooth))
			
			muobs_inter_norm = muobs_det_norm[ind_peaks_neg[pp]+tt:ind_peaks[pp+1]-tt]
			muobs_inter_norm_smooth = csaps(tobs_inter_tmp, muobs_inter_norm, tobs_inter_tmp, smooth=smoothes[cc])
			muobs_inter_norm_smooth_err = np.concatenate((muobs_inter_norm_smooth_err,muobs_inter_norm-muobs_inter_norm_smooth))
			
			ShearStrainobs_inter_norm = ShearStrainobs_det_norm[ind_peaks_neg[pp]+tt:ind_peaks[pp+1]-tt]
			ShearStrainobs_inter_norm_smooth = csaps(tobs_inter_tmp, ShearStrainobs_inter_norm, tobs_inter_tmp, smooth=smoothes[cc])
			ShearStrainobs_inter_norm_smooth_err = np.concatenate((ShearStrainobs_inter_norm_smooth_err,ShearStrainobs_inter_norm-ShearStrainobs_inter_norm_smooth))
			try:
				ecDisp_inter = np.concatenate((ecDisp_inter,ecDispobs[ind_peaks_neg[pp]+tt:ind_peaks[pp+1]-tt]))
			except:
				import pdb; pdb.set_trace()
		#import pdb; pdb.set_trace()
		if exp_name == 'i417':
			p, C_p = np.polyfit(tobs_inter, ecDisp_inter*1e-3, deg=1, cov=True)
		else:
			p, C_p = np.polyfit(tobs_inter, ecDisp_inter*1e3, deg=1, cov=True)
		v0_inter[cc] = p[0]
		std_v0_inter[cc] = np.sqrt(C_p[0,0])
		
		std_sigman_inter[cc] = np.std(NormStressobs_inter_smooth_err)
		std_tauf_inter[cc] = np.std(ShearStressobs_inter_smooth_err)
		
		std_NormStress_inter_norm_smooth[cc] = np.std(NormStressobs_inter_norm_smooth_err)
		std_ShearStress_inter_norm_smooth[cc] = np.std(ShearStressobs_inter_norm_smooth_err)
		std_LayerThick_inter_norm_smooth[cc] = np.std(LayerThickobs_inter_norm_smooth_err)
		std_ecDisp_inter_norm_smooth[cc] = np.std(ecDispobs_inter_norm_smooth_err)
		std_mu_inter_norm_smooth[cc] = np.std(muobs_inter_norm_smooth_err)
		std_ShearStrain_inter_norm_smooth[cc] = np.std(ShearStrainobs_inter_norm_smooth_err)
		
		exp_names.append(exp_name)
		
		#W_sigman[cc],pvalue_sigman[cc] = shapiro(NormStressobs_err)
		W_sigman[cc],pvalue_sigman[cc] = shapiro(NormStressobs_inter_smooth_err)
		if pvalue_sigman[cc]<0.05:
			print('H0 (sigman_err drawn from normal distribution) rejected')
			flag_rejected_sigman = True
		else:
			print('H0 (sigman_err drawn from normal distribution) not rejected')
			flag_rejected_sigman = False
		
		#W_tauf[cc],pvalue_tauf[cc] = shapiro(ShearStressobs_err)
		W_tauf[cc],pvalue_tauf[cc] = shapiro(ShearStressobs_inter_smooth_err)
		if pvalue_tauf[cc]<0.05:
			print('H0 (tauf_err drawn from normal distribution) rejected')
			flag_rejected_tauf = True
		else:
			print('H0 (tauf_err drawn from normal distribution) not rejected')
			flag_rejected_tauf = False

TabS2 = [exp_names,
		 tstart,
		 tend,
		 sigman0,
		 std_sigman0,
		 v0,
		 std_v0]
TabS3 = [exp_names,
		 std_ShearStress_inter_norm_smooth,
		 std_NormStress_inter_norm_smooth,
		 std_LayerThick_inter_norm_smooth,
		 std_ecDisp_inter_norm_smooth,
		 std_mu_inter_norm_smooth,
		 std_ShearStrain_inter_norm_smooth]