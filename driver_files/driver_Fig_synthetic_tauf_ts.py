#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 17:42:23 2021

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
import numpy as np
from scipy.signal import find_peaks
from csaps import csaps

import matplotlib
matplotlib.rcParams['font.family'] = 'times'

plt.close('all')
tic_begin = time.time()

dirs = {'main' : os.getcwd(),
		'data' : '/Users/vinco/Work/Data/labquakesde/'
		}
			
#%% SELECT CASE STUDY AND SET UP DIRECTORIES
diff_eq_type = 'param4'
#diff_eq_type = 'param5'
#diff_eq_type = 'param6'
#diff_eq_type = 'param7'

Anoise = 0
#Anoise = 4e3

output_dir = dirs['main'] + \
 '/../scenarios/synthetic/'+diff_eq_type+'/_figures/'

if Anoise == 0:
	file_name = 'Fig_tauf_ts_noisefree.png'
else:
	file_name = 'Fig_tauf_ts_noiseobs.png'

output_file = output_dir + file_name

case_studies = ['synthetic/'+diff_eq_type+'/b726',\
				'synthetic/'+diff_eq_type+'/b698',\
				'synthetic/'+diff_eq_type+'/b728',\
				'synthetic/'+diff_eq_type+'/b721',\
				'synthetic/'+diff_eq_type+'/i417']
sim_case = ['sim 4', 'sim 8', 'sim 10', 'sim 11', 'sim 14']

#%% SET FLAGS
flag_save_fig     = False
flag_load_data    = True

dpi = 300

#%% SET PARAMETERS
Ncases = len(case_studies)

v0 = 10e-6
L = 3e-6
Dt = 200

smoothes = [0.9]*Ncases

sigman0 = [0.0]*Ncases

min_dist_peak = 200

markersize = 15

alpha = np.zeros((Ncases,1))
sigman0 = np.zeros((Ncases,1))
sigman0_nominal = [0]*len(case_studies)
epsilon_NormStress = np.zeros((Ncases,1))
epsilon_ShearStress = np.zeros((Ncases,1))

format_color = ['.-k', '.-b', '.-r', '.-c', '.-g', '.-m', '.-y', \
				'sk', 'sb', 'sr', 'sc', 'sg', 'sm', 'sy']

figx = 15
figy = 9
fontsize = 24

if diff_eq_type == 'param4':
	title_string1 = r'ODE periodic'
elif diff_eq_type == 'param5':
	title_string1 = r'ODE deterministic chaos'
elif diff_eq_type == 'param6':
	title_string1 = r'SDE$_{yu}$'
elif diff_eq_type == 'param7':
	title_string1 = r'SDE$_{yu}$'

if Anoise == 0:
	title_string2 = ''
else:
	title_string2 = r' + $\epsilon_{\tau_f}$'
title_string = title_string1 + title_string2

print("")
plt.figure(figsize=(figx, figy))
plt.grid()
ax = plt.subplot(111)
for cc,case_study in enumerate(case_studies):
	exp_name = case_study[-4:]
	dirs['pickles'] = dirs['main']+'/../scenarios/'+case_study+'/pickles/'
	dirs['figures'] = dirs['main']+'/../scenarios/'+case_study+'/figures/'
	
	print(case_study)
	
	#% LOAD DATA
	if flag_load_data==True:
		sol_t      = np.loadtxt(dirs['data']+case_study+'/sol_t.txt')
		sol_u      = np.loadtxt(dirs['data']+case_study+'/sol_u.txt')
		a          = np.loadtxt(dirs['data']+case_study+'/a.txt')
		sigman0_cc = np.loadtxt(dirs['data']+case_study+'/sigman0.txt')
		v0         = np.loadtxt(dirs['data']+case_study+'/v0.txt')
		L1         = np.loadtxt(dirs['data']+case_study+'/L1.txt')
		tau0       = np.loadtxt(dirs['data']+case_study+'/tau0.txt')
		mu0        = tau0/sigman0_cc
		DT = Dt*v0/L1
		Tend = sol_t[-1]
		Tmin = Tend - DT
		tmin = Tmin*L1/v0
		
		sigman0_nominal[cc] = sigman0_cc/1e6
		
		#% OBSERVED DATA
		tobs = sol_t[sol_t>Tmin]*L1/v0
		tobs = tobs - tobs[0]
		Nt = tobs.shape[0]
		vobs = v0*np.exp(sol_u[sol_t>Tmin,0])
		ShearStressobs = np.zeros((Nt,))
		ShearStressobs = (tau0 + a*sigman0_cc*sol_u[sol_t>Tmin,1] + \
					Anoise*np.random.randn(Nt))/1e6
		NormStressobs = sigman0_nominal[cc] + \
			(sigman0_nominal[cc]*a/mu0)*sol_u[sol_t>Tmin,2]
		
		dtobs = tobs[1]-tobs[0]
		
		p = np.polyfit(tobs, ShearStressobs, deg=1)
		ShearStressobs_det = ShearStressobs - (p[0]*tobs + p[1])
		ShearStressobs0 = p[1]
		
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
		
		
		# PEAKS CALCULATION
		ind_peaks = find_peaks(X[:,0],distance=min_dist_peak)[0]
		ind_peaks_neg = find_peaks(-X[:,0],distance=min_dist_peak)[0]
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
			twin = np.min([min_dist_peak,ind_peaks[pp]])
			ind_peaks[pp] = ind_peaks[pp]-twin + \
				np.argmax(Xobs[ind_peaks[pp]-twin:ind_peaks[pp]+twin,0])
			twin_neg = np.min([min_dist_peak,ind_peaks_neg[pp]])
			ind_peaks_neg[pp] = ind_peaks_neg[pp]-twin_neg + \
				np.argmax(-Xobs[ind_peaks_neg[pp]-twin_neg : \
					ind_peaks_neg[pp]+twin_neg,0])
		
		
		ind_peaks = np.unique(ind_peaks)
		ind_peaks_neg = np.unique(ind_peaks_neg)
		
		Npeaks = ind_peaks.shape[0]
		ind_peaks_center = np.zeros((Npeaks,))
		ind_peaks_center = ind_peaks_center.astype(int)
		for pp in np.arange(Npeaks):
			ind_peaks_center[pp] = int(ind_peaks[pp] + \
				  np.argmin(np.abs(Xobs[ind_peaks[pp]:ind_peaks_neg[pp]+1,0])))
		
		
		ind_inter = np.abs(np.gradient(Xobs[:,0]))<=0.01
		sigman0[cc] = np.mean(NormStressobs[ind_inter])
		epsilon_NormStress[cc] = np.std(NormStressobs[ind_inter]-sigman0[cc])
		ShearStress_smooth = csaps(tobs, ShearStressobs, tobs,
							 smooth=smoothes[cc])
		epsilon_ShearStress[cc] = np.std(ShearStressobs[ind_inter] - \
								   ShearStress_smooth[ind_inter])
		
		
		plt.plot(tobs, ShearStressobs0 + ShearStressobs_det, '.-k')
		plt.plot(tobs[ind_peaks], \
		   ShearStressobs0 + ShearStressobs_det[ind_peaks], '.', c='#7570b3',
		   markersize=markersize)
		plt.plot(tobs[ind_peaks_neg], \
		   ShearStressobs0 + ShearStressobs_det[ind_peaks_neg], '.',
		   c='#1b9e77', markersize=markersize)
		plt.plot(tobs[ind_peaks_center], \
		   ShearStressobs0 + ShearStressobs_det[ind_peaks_center], '.',
			   c='#d95f02', markersize=markersize)
		#plt.text(tobs[0]-10, 0.25+np.mean(ShearStressobs0 + \
		#							ShearStressobs_det), exp_name, fontsize=18)
		plt.text(tobs[0]-10, 0.05+np.max(ShearStressobs0 + ShearStressobs_det),
			sim_case[cc], fontsize=18)
plt.suptitle(title_string, x=0.51, y=0.97, fontsize=28)
plt.xlabel('Time [s]', fontsize=fontsize)
plt.ylabel(r'$\tau_f$ [MPa]', fontsize=fontsize)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
x_lettera = 0.96
y_lettera = 0.95
x1lim = ax.get_xlim()
y1lim = ax.get_ylim()
ax.text(x1lim[0]+x_lettera*(x1lim[1]-x1lim[0]), \
		y1lim[0]+y_lettera*(y1lim[1]-y1lim[0]),'(a)', fontsize=fontsize)
if flag_save_fig==True:
	plt.savefig(output_file, dpi=dpi, bbox_inches='tight')

