#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 19:09:31 2021

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
from utils.load_utils import import_data, load_pickle
from utils.param_utils import set_param
import numpy as np
from scipy.signal import find_peaks
from csaps import csaps
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import gridspec
from matplotlib import cm
import matplotlib
from matplotlib import ticker


plt.close('all')
tic_begin = time.time()

dirs = {'main' : os.getcwd(),
		'data' : '/Users/vinco/Work/Data/labquakesde/'
		}
			
#%% SELECT CASE STUDY AND SET UP DIRECTORIES
output_dir = dirs['main'] + \
 '/../scenarios/labquakes/MeleVeeduetal2020/_figures/'

file_name_tauf = 'Fig_Lorenz_map_Dtau_f'
file_name_T = 'Fig_Lorenz_map_T'

output_file_tauf = output_dir + file_name_tauf
output_file_T = output_dir + file_name_T

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
flag_save_fig     = False 
flag_load_data    = True
flag_load_results = True
flag_load_EVT     = True
flag_load_LEs     = True

dpi = 300

fontsize_suptitle = 22
fontsize_title = 16
fontsize_axes = 16 #12
fontsize_legend = 11 #9.5
fontsize = 16
format_color = ['ok', 'ob', 'or', 'oc', 'og', 'om', 'oy', \
				'sk', 'sb', 'sr', 'sc', 'sg', 'sm', 'sy']
	
@ticker.FuncFormatter
def major_formatter(x, pos):
    return f'{x:.2f}'
	
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
tt=-1
for mm in min_dist:
	tt+=1
	tau_delay.append(min_dist[tt]-1)
Ntau_delay = len(tau_delay)

smoothes = [0.9]*Ncases

sigman0 = [0.0]*Ncases

min_dist_peak = 200


sigman0 = np.zeros((Ncases,1))
sigman0_nominal = np.zeros((Ncases,1))
epsilon_NormStress = np.zeros((Ncases,1))
epsilon_ShearStress = np.zeros((Ncases,1))
Tpre = dict()
Tnext = dict()
Tprepost = dict()
Dtau_f = dict()
tauf_center = dict()
tauf_max_lab = dict()
tauf_min_lab = dict()
t_max_lab = dict()
t_min_lab = dict()
print("")
for cc,case_study in enumerate(case_studies):
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
		ShearStressobs = data['ShearStress']
		NormStressobs = data['NormStress']
		
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
		
		# CV CALCULATION
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
		
		#dT = tobs[ind_peaks[1:]]-tobs[ind_peaks[:-1]]
		dT = tobs[ind_peaks_center[1:]]-tobs[ind_peaks_center[:-1]]
		dT_mean = np.mean(dT)
		dT_std = np.std(dT)
		
		ind_inter = np.abs(np.gradient(Xobs[:,0]))<=0.01
		sigman0[cc] = np.mean(NormStressobs[ind_inter])
		epsilon_NormStress[cc] = np.std(NormStressobs[ind_inter]-sigman0[cc])
		ShearStress_smooth = csaps(tobs, ShearStressobs, tobs,
							 smooth=smoothes[cc])
		epsilon_ShearStress[cc] = np.std(ShearStressobs[ind_inter] - \
								   ShearStress_smooth[ind_inter])
		
		Dtau_f[case_study] = ShearStressobs_det[ind_peaks] - \
			ShearStressobs_det[ind_peaks_neg]
		
		Tpre[case_study] = tobs[ind_peaks_center[1:-1]] - \
			tobs[ind_peaks_center[:-2]]
		Tnext[case_study] = tobs[ind_peaks_center[2:]] - \
			tobs[ind_peaks_center[1:-1]]
		
		tauf_center[case_study] = ShearStressobs0 + \
			ShearStressobs_det[ind_peaks_center]
		tauf_max_lab[case_study] = ShearStressobs0 + \
			ShearStressobs_det[ind_peaks]
		tauf_min_lab[case_study] = ShearStressobs0 + \
			ShearStressobs_det[ind_peaks_neg]
		t_max_lab[case_study] = tobs[ind_peaks]
		t_min_lab[case_study] = tobs[ind_peaks_neg]

			
#%% SELECT CASE STUDY AND SET UP DIRECTORIES
#diff_eq_type = 'param4' # ODE periodic
#diff_eq_type = 'param5' # ODE chaotic
#diff_eq_type = 'param6' # SDE

diff_eq_types = ['param4', 'param5', 'param6']

Anoises = [0, 4e3]

for eqt,diff_eq_type in enumerate(diff_eq_types):
	
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
	
	for an,Anoise in enumerate(Anoises):
		
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
		tt=-1
		for mm in min_dist:
			tt+=1
			tau_delay.append(min_dist[tt]-1)
		Ntau_delay = len(tau_delay)
		
		smoothes = [0.9]*Ncases
		
		sigman0 = [0.0]*Ncases
		
		min_dist_peak = 200
		
		
		sigman0 = np.zeros((Ncases,1))
		sigman0_nominal = np.zeros((Ncases,1))
		epsilon_NormStress = np.zeros((Ncases,1))
		epsilon_ShearStress = np.zeros((Ncases,1))
		print("")
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
				
				sigman0[cc] = np.mean(NormStressobs)
				
				Xobs = np.array([ShearStressobs_det]).T
				Nt,Nx = Xobs.shape
				X = np.zeros((Nt,Nx))
				for nn in np.arange(Nx):
					X[:,nn] = csaps(tobs, Xobs[:,nn], tobs, smooth=smoothes[nn])
				X_norm = np.zeros((Nt,Nx))
				for ii in np.arange(Nx):
					X_norm[:,ii] = (X[:,ii]-np.min(X[:,ii])) / \
						(np.max(X[:,ii])-np.min(X[:,ii]))
				
				# CV CALCULATION
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
				
				#dT = tobs[ind_peaks[1:]]-tobs[ind_peaks[:-1]]
				dT = tobs[ind_peaks_center[1:]]-tobs[ind_peaks_center[:-1]]
				dT_mean = np.mean(dT)
				dT_std = np.std(dT)
				
				ind_inter = np.abs(np.gradient(Xobs[:,0]))<=0.01
				sigman0[cc] = np.mean(NormStressobs[ind_inter])
				epsilon_NormStress[cc] = np.std(NormStressobs[ind_inter]-sigman0[cc])
				ShearStress_smooth = csaps(tobs, ShearStressobs, tobs,
									 smooth=smoothes[cc])
				epsilon_ShearStress[cc] = np.std(ShearStressobs[ind_inter] - \
										   ShearStress_smooth[ind_inter])
				
				Dtau_f[case_study+'_'+str(Anoise)] = ShearStressobs_det[ind_peaks] - \
					ShearStressobs_det[ind_peaks_neg]
				
				Tpre[case_study+'_'+str(Anoise)] = tobs[ind_peaks_center[1:-1]] - \
					tobs[ind_peaks_center[:-2]]
				Tnext[case_study+'_'+str(Anoise)] = tobs[ind_peaks_center[2:]] - \
					tobs[ind_peaks_center[1:-1]]
				
				tauf_center[case_study] = ShearStressobs0 + \
					ShearStressobs_det[ind_peaks_center]
				
		
		
#%%  
fig = plt.figure(figsize=(15,8))

diff_eq_types = ['param6', 'param5', 'param4']

Anoises = [0]

color_sims = ['#7570b3', '#d95f02', '#1b9e77']
ls_sims = ['o', 's']

for eqt,diff_eq_type in enumerate(diff_eq_types):
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
	for an,Anoise in enumerate(Anoises):
		print(diff_eq_type+' '+str(Anoise))
		for cc,case_study in enumerate(case_studies):
			exp_name = case_study[-4:]
			ax = plt.subplot(3,5,cc+1)
			plt.scatter(Dtau_f[case_study+'_'+str(Anoise)][:-1],Dtau_f[case_study+'_'+str(Anoise)][1:],
						  50, marker=ls_sims[an], c=color_sims[eqt])

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
for cc,case_study in enumerate(case_studies):
	exp_name = case_study[-4:]
	ax = plt.subplot(3,5,cc+1)
	plt.title(exp_name, fontsize=fontsize_title)
	plt.scatter(Dtau_f[case_study][:-1],Dtau_f[case_study][1:],
		  50, facecolors='none', edgecolors='k')
	if np.mod(cc,5)==0:
		plt.ylabel(r'$\Delta\tau_f^{i+1}$ [MPa]', fontsize=fontsize_axes)
	if cc+1>=10:
		plt.xlabel(r'$\Delta\tau_f^{i}$ [MPa]', fontsize=fontsize_axes)
	plt.tick_params(axis = 'both', which = 'major', labelsize=fontsize_axes)
	plt.grid()
	ax.set_aspect('equal','box')

plt.suptitle(r'Cobweb plot of $\Delta\tau_f$ for simulations without observational noise',
			 fontsize=fontsize_suptitle)
ax = plt.subplot(3,5,15)
ax.set_xticks([])
ax.set_yticks([])
plt.axis('off')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
plt.scatter(0.1, 0.65, 50, facecolors='none', edgecolors='k')
plt.scatter(0.1, 0.55, 50, marker=ls_sims[0], c=color_sims[2])
plt.scatter(0.1, 0.45, 50, marker=ls_sims[0], c=color_sims[1])
plt.scatter(0.1, 0.35, 50, marker=ls_sims[0], c=color_sims[0])
plt.text(0.15, 0.625, 'Laboratory', fontsize=fontsize_legend)
plt.text(0.15, 0.525, 'ODE (periodic)', fontsize=fontsize_legend)
plt.text(0.15, 0.425, 'ODE (chaotic)', fontsize=fontsize_legend)
plt.text(0.15, 0.325, 'SDE', fontsize=fontsize_legend)
ax.set_aspect('equal','box')
plt.tight_layout()

if flag_save_fig==True:
	plt.savefig(output_file_tauf+'_noisefree.png', dpi=dpi,
		  bbox_inches='tight')
	
#%%
fig = plt.figure(figsize=(15,8))

diff_eq_types = ['param6', 'param5', 'param4']

Anoises = [4e3]

color_sims = ['#7570b3', '#d95f02', '#1b9e77']
ls_sims = ['o', 's']

for eqt,diff_eq_type in enumerate(diff_eq_types):
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
	for an,Anoise in enumerate(Anoises):
		print(diff_eq_type+' '+str(Anoise))
		for cc,case_study in enumerate(case_studies):
			exp_name = case_study[-4:]
			ax = plt.subplot(3,5,cc+1)
			plt.scatter(Dtau_f[case_study+'_'+str(Anoise)][:-1],Dtau_f[case_study+'_'+str(Anoise)][1:],
						  50, marker=ls_sims[an], c=color_sims[eqt])

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
for cc,case_study in enumerate(case_studies):
	exp_name = case_study[-4:]
	ax = plt.subplot(3,5,cc+1)
	plt.title(exp_name, fontsize=fontsize_title)
	plt.scatter(Dtau_f[case_study][:-1],Dtau_f[case_study][1:],
			  50, facecolors='none', edgecolors='k')
	if np.mod(cc,5)==0:
		plt.ylabel(r'$\Delta\tau_f^{i+1}$ [MPa]', fontsize=fontsize_axes)
	if cc+1>=10:
		plt.xlabel(r'$\Delta\tau_f^{i}$ [MPa]', fontsize=fontsize_axes)
	plt.tick_params(axis = 'both', which = 'major', labelsize=fontsize_axes)
	plt.grid()
	ax.set_aspect('equal','box')

plt.suptitle(r'Cobweb plot of $\Delta\tau_f$ for simulations with also observational noise',
			 fontsize=fontsize_suptitle)

ax = plt.subplot(3,5,15)
ax.set_xticks([])
ax.set_yticks([])
plt.axis('off')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
plt.scatter(0.1, 0.65, 50, facecolors='none', edgecolors='k')
plt.scatter(0.1, 0.55, 50, marker=ls_sims[0], c=color_sims[2])
plt.scatter(0.1, 0.45, 50, marker=ls_sims[0], c=color_sims[1])
plt.scatter(0.1, 0.35, 50, marker=ls_sims[0], c=color_sims[0])
plt.text(0.15, 0.625, 'Laboratory', fontsize=fontsize_legend)
plt.text(0.15, 0.525, 'ODE (periodic)', fontsize=fontsize_legend)
plt.text(0.15, 0.425, 'ODE (chaotic)', fontsize=fontsize_legend)
plt.text(0.15, 0.325, 'SDE', fontsize=fontsize_legend)
ax.set_aspect('equal','box')
plt.tight_layout()

if flag_save_fig==True:
	plt.savefig(output_file_tauf+'_noiseobs.png', dpi=dpi,
		  bbox_inches='tight')
	
#%%
fig = plt.figure(figsize=(15,8))

diff_eq_types = ['param6', 'param5', 'param4']

Anoises = [0]

color_sims = ['#7570b3', '#d95f02', '#1b9e77']
ls_sims = ['o', 's']

for eqt,diff_eq_type in enumerate(diff_eq_types):
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
	for an,Anoise in enumerate(Anoises):
		print(diff_eq_type+' '+str(Anoise))
		for cc,case_study in enumerate(case_studies):
			exp_name = case_study[-4:]
			ax = plt.subplot(3,5,cc+1)
			plt.scatter(Tpre[case_study+'_'+str(Anoise)][:-1],
			   Tnext[case_study+'_'+str(Anoise)][1:],
			   50, marker=ls_sims[an], c=color_sims[eqt])

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
for cc,case_study in enumerate(case_studies):
	exp_name = case_study[-4:]
	ax = plt.subplot(3,5,cc+1)
	plt.title(exp_name, fontsize=fontsize_title)
	plt.scatter(Tpre[case_study][:-1], Tnext[case_study][1:],
		  50, facecolors='none', edgecolors='k')
	if np.mod(cc,5)==0:
		plt.ylabel(r'$T_{next}$ [s]', fontsize=fontsize_axes)
	if cc+1>=10:
		plt.xlabel(r'$T_{pre}$ [s]', fontsize=fontsize_axes)
	plt.tick_params(axis = 'both', which = 'major', labelsize=fontsize_axes)
	plt.grid()
	ax.set_aspect('equal','box')

plt.suptitle(r'Cobweb plot of $T$ for simulations without observational noise',
			 fontsize=fontsize_suptitle)
ax = plt.subplot(3,5,15)
ax.set_xticks([])
ax.set_yticks([])
plt.axis('off')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
plt.scatter(0.1, 0.65, 50, facecolors='none', edgecolors='k')
plt.scatter(0.1, 0.55, 50, marker=ls_sims[0], c=color_sims[2])
plt.scatter(0.1, 0.45, 50, marker=ls_sims[0], c=color_sims[1])
plt.scatter(0.1, 0.35, 50, marker=ls_sims[0], c=color_sims[0])
plt.text(0.15, 0.625, 'Laboratory', fontsize=fontsize_legend)
plt.text(0.15, 0.525, 'ODE (periodic)', fontsize=fontsize_legend)
plt.text(0.15, 0.425, 'ODE (chaotic)', fontsize=fontsize_legend)
plt.text(0.15, 0.325, 'SDE', fontsize=fontsize_legend)
ax.set_aspect('equal','box')
plt.tight_layout()

if flag_save_fig==True:
	plt.savefig(output_file_T+'_noisefree.png', dpi=dpi,
		  bbox_inches='tight')

#%%
fig = plt.figure(figsize=(15,8))

diff_eq_types = ['param6', 'param5', 'param4']

Anoises = [4e3]

color_sims = ['#7570b3', '#d95f02', '#1b9e77']
ls_sims = ['o', 's']

for eqt,diff_eq_type in enumerate(diff_eq_types):
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
	for an,Anoise in enumerate(Anoises):
		print(diff_eq_type+' '+str(Anoise))
		for cc,case_study in enumerate(case_studies):
			exp_name = case_study[-4:]
			ax = plt.subplot(3,5,cc+1)
			plt.scatter(Tpre[case_study+'_'+str(Anoise)][:-1],
			   Tnext[case_study+'_'+str(Anoise)][1:],
			   50, marker=ls_sims[an], c=color_sims[eqt])

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
for cc,case_study in enumerate(case_studies):
	exp_name = case_study[-4:]
	ax = plt.subplot(3,5,cc+1)
	plt.title(exp_name, fontsize=fontsize_title)
	plt.scatter(Tpre[case_study][:-1], Tnext[case_study][1:],
			  50, facecolors='none', edgecolors='k')
	if np.mod(cc,5)==0:
		plt.ylabel(r'$T_{next}$ [s]', fontsize=fontsize_axes)
	if cc+1>=10:
		plt.xlabel(r'$T_{pre}$ [s]', fontsize=fontsize_axes)
	plt.tick_params(axis = 'both', which = 'major', labelsize=fontsize_axes)
	plt.grid()
	ax.set_aspect('equal','box')

plt.suptitle(r'Cobweb plot of $T$ for simulations with also observational noise',
			 fontsize=fontsize_suptitle)

ax = plt.subplot(3,5,15)
ax.set_xticks([])
ax.set_yticks([])
plt.axis('off')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
plt.scatter(0.1, 0.65, 50, facecolors='none', edgecolors='k')
plt.scatter(0.1, 0.55, 50, marker=ls_sims[0], c=color_sims[2])
plt.scatter(0.1, 0.45, 50, marker=ls_sims[0], c=color_sims[1])
plt.scatter(0.1, 0.35, 50, marker=ls_sims[0], c=color_sims[0])
plt.text(0.15, 0.625, 'Laboratory', fontsize=fontsize_legend)
plt.text(0.15, 0.525, 'ODE (periodic)', fontsize=fontsize_legend)
plt.text(0.15, 0.425, 'ODE (chaotic)', fontsize=fontsize_legend)
plt.text(0.15, 0.325, 'SDE', fontsize=fontsize_legend)
ax.set_aspect('equal','box')
plt.tight_layout()

if flag_save_fig==True:
	plt.savefig(output_file_T+'_noiseobs.png', dpi=dpi,
		  bbox_inches='tight')