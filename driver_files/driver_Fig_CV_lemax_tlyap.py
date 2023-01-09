#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 09:42:09 2021

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
from matplotlib import gridspec

plt.rcParams.update({'font.size': 22})

plt.close('all')
tic_begin = time.time()

dirs = {'main' : os.getcwd(),
		'data' : '/Users/vinco/Work/Data/labquakesde/'
		}


#%% SET FLAGS
flag_save_fig     = False
flag_load_data    = True
flag_load_results = True
flag_load_LEs     = True

dpi = 300		

#%%
id_name = []

CV = dict()
Tpre = dict()
Tnext = dict()
Tprepost = dict()
Dtau_f = dict()
tauf_center = dict()

#%%
output_dir = dirs['main'] + \
 '/../scenarios/labquakes/MeleVeeduetal2020/_figures/'

file_name = 'Fig_CV_lemax_tlyap'

output_file = output_dir + file_name

#%%
diff_eq_types = ['param4', 'param6']

Anoises = [0, 4e3]

for Anoise in Anoises:
	for diff_eq_type in diff_eq_types:
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
		case_studies_noise = ['synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b724', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b722', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b696', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b726', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b694', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b697', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b693', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b698', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b695', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b728', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b721', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b725', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b727', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/i417']
			
		
		
		
		#%% SET PARAMETERS
		cc = -1
		Ncases = len(case_studies)
		
		v0 = 10e-6
		L = 3e-6
		Dt = 200
		#Tmin = 0.5*(1000.0+1000.0/3) #1000
		
		# Embedding dimensions to test
		m_list = [3,4,5,6,7,8,9,10,11,12,13,14,15,20,50,100] #[2,3,4,5,6,7,8]
		#m_list = [3,4,5]
		Nm = len(m_list)
		
		# Minimum distance between peaks (add 1 for scipy notation), i.e. if you want
		# a minimum distance of 3 use 4
		min_dist = [2,3,4,5,6,7,8,9,10,15,20] #[3,4,5,6,7,8,9,10,15,20,25,30,35]
		#min_dist = [2,3,4,5]
		Nmin_dist = len(min_dist)
		
		smoothes = [0.9]*Ncases
		
		sigman0 = [0.0]*Ncases
		
		alpha = np.zeros((Ncases,1))
		sigman0 = np.zeros((Ncases,1))
		sigman0_nominal = [0]*len(case_studies)
		epsilon_NormStress = np.zeros((Ncases,1))
		epsilon_ShearStress = np.zeros((Ncases,1))
		
		figx = 13.5
		figy = 8
		figx_tauf = 13.5
		figy_tauf = 3
		
		min_dist = 10
		Twin = 200
		
		D1 = np.zeros((Ncases,Nm,Nmin_dist))
		for case_study in case_studies:
			exp_name = case_study[-5:]
			dirs['pickles'] = dirs['main']+'/../scenarios/'+case_study+'/pickles/'
			dirs['figures'] = dirs['main']+'/../scenarios/'+case_study+'/figures/'
			print("")
			print("")
			print(case_study)
			cc+=1
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
				ShearStressobs = (tau0 + a*sigman0_cc*sol_u[sol_t>Tmin,1] + Anoise*np.random.randn(Nt))/1e6
				NormStressobs = sigman0_nominal[cc] + (sigman0_nominal[cc]*a/mu0)*sol_u[sol_t>Tmin,2]
				
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
					X_norm[:,ii] = (X[:,ii]-np.min(X[:,ii])) / (np.max(X[:,ii])-np.min(X[:,ii]))
				
				
				# CV CALCULATION
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
				
				dT = tobs[ind_peaks_center[1:]]-tobs[ind_peaks_center[:-1]]
				dT_mean = np.mean(dT)
				dT_std = np.std(dT)
				CV[case_studies_noise[cc]] = dT_std/dT_mean
				
				ind_inter = np.abs(np.gradient(Xobs[:,0]))<=0.01
				sigman0[cc] = np.mean(NormStressobs[ind_inter])
				epsilon_NormStress[cc] = np.std(NormStressobs[ind_inter]-sigman0[cc])
				ShearStress_smooth = csaps(tobs, ShearStressobs, tobs, smooth=smoothes[cc])
				epsilon_ShearStress[cc] = np.std(ShearStressobs[ind_inter]-ShearStress_smooth[ind_inter])
				
				Dtau_f[case_studies_noise[cc]] = ShearStressobs_det[ind_peaks]-ShearStressobs_det[ind_peaks_neg]
				
				Tpre[case_studies_noise[cc]] = tobs[ind_peaks_center[1:-1]]-tobs[ind_peaks_center[:-2]]
				Tnext[case_studies_noise[cc]] = tobs[ind_peaks_center[2:]]-tobs[ind_peaks_center[1:-1]]
				Tprepost[case_studies_noise[cc]] = Tpre[case_studies_noise[cc]]/Tnext[case_studies_noise[cc]]
				
				id_name.append(case_studies_noise[cc])

#%%
if flag_load_results==True:
	if flag_load_LEs==True:
		LEs_mean = dict()
		LEs_std  = dict()
		
for Anoise in Anoises:
	for diff_eq_type in diff_eq_types:
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
		case_studies_noise = ['synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b724', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b722', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b696', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b726', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b694', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b697', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b693', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b698', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b695', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b728', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b721', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b725', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b727', \
							'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/i417']
			
		for cc,case_study in enumerate(case_studies):
			dirs['pickles'] = dirs['main']+'/../scenarios/'+case_study+'/pickles/'
			print("")
			print(case_study)
			smoothes = np.zeros((len(case_studies),1))
			if flag_load_LEs==True:
				mhat_tmp, tau_delay_tmp, LEs_tmp, LEs_mean_tmp, LEs_std_tmp, \
					eps_over_L_tmp = load_pickle(\
				    dirs['pickles']+'/LEs_ShearStress_det_'+str('%.2e' %(Anoise))+'.pickle', \
				    message='Loading pickled variables: LEs')
				
				LEs_mean[case_studies_noise[cc]] = LEs_mean_tmp[-1,:]
				LEs_std[case_studies_noise[cc]] = LEs_std_tmp[-1,:]

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


#%% SET PARAMETERS

Ncases = len(case_studies)

smoothes = [0.9]*Ncases

sigman0 = [0.0]*Ncases

min_dist_peak = 45
Twin = 200

sigman0 = np.zeros((Ncases,1))
epsilon_NormStress = np.zeros((Ncases,1))
epsilon_ShearStress = np.zeros((Ncases,1))
print("")
for cc,case_study in enumerate(case_studies):
	
	dirs['pickles'] = dirs['main']+'/../scenarios/'+case_study+'/pickles/'
	dirs['figures'] = dirs['main']+'/../scenarios/'+case_study+'/figures/'
	print(case_study)
	
	#% LOAD DATA
	exp_name = case_study[-4:]
	filename = dirs['data']+case_study+'/'+exp_name+'.txt'
	parameters = set_param(case_study)
	
	if flag_load_data==True:
		data = import_data(dirs,filename,parameters,\
					 struct='MeleVeeduetal2020', file_format='txt')
		
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
		
		p = np.polyfit(tobs, LayerThickobs, deg=2)
		LayerThickobs_det = LayerThickobs - (p[0]*tobs**2 + p[1]*tobs + p[2])
		
		p = np.polyfit(tobs, ecDispobs, deg=1)
		ecDispobs_det = ecDispobs - (p[0]*tobs + p[1])
		
		p = np.polyfit(tobs, ShearStressobs, deg=1)
		ShearStressobs_det = ShearStressobs - (p[0]*tobs + p[1])
		ShearStressobs0 = p[1]
		
		p = np.polyfit(tobs, muobs, deg=1)
		muobs_det = muobs - (p[0]*tobs + p[1])
		
		p = np.polyfit(tobs, ShearStrainobs, deg=3)
		ShearStrainobs_det = ShearStrainobs - (p[0]*tobs**3 + p[1]*tobs**2 + p[2]*tobs + p[3])
		#ShearStrainobs_det = np.random.randn(tobs.shape[0])
		
		del p
		
		sigman0[cc] = np.mean(NormStressobs)
		
		Xobs = np.array([ShearStressobs_det]).T
		Nt,Nx = Xobs.shape
		X = np.zeros((Nt,Nx))
		for nn in np.arange(Nx):
			X[:,nn] = csaps(tobs, Xobs[:,nn], tobs, smooth=smoothes[nn])
		X_norm = np.zeros((Nt,Nx))
		for ii in np.arange(Nx):
			X_norm[:,ii] = (X[:,ii]-np.min(X[:,ii])) / (np.max(X[:,ii])-np.min(X[:,ii]))
		
		
		# CV CALCULATION
		if exp_name == 'b724':
			min_dist_peak = 10
		else:
			min_dist_peak = 45
		ind_peaks = find_peaks(X[:,0],distance=min_dist_peak)[0]
		
		dT_smooth = tobs[ind_peaks[1:]]-tobs[ind_peaks[:-1]]
		dT_smooth_mean = np.mean(dT_smooth)
		dT_smooth_std = np.std(dT_smooth)
		
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
			twin = min_dist_peak
			ind_peaks[pp] = ind_peaks[pp]-twin+np.argmax(Xobs[ind_peaks[pp]-twin:ind_peaks[pp]+twin,0])
			twin_neg = min_dist_peak
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
		
		dT = tobs[ind_peaks_center[1:]]-tobs[ind_peaks_center[:-1]]
		dT_mean = np.mean(dT)
		dT_std = np.std(dT)
		CV[case_study] = dT_std/dT_mean
		
		ind_inter = np.abs(np.gradient(Xobs[:,0]))<=0.01
		sigman0[cc] = np.mean(NormStressobs[ind_inter])
		epsilon_NormStress[cc] = np.std(NormStressobs[ind_inter]-sigman0[cc])
		ShearStress_smooth = csaps(tobs, ShearStressobs, tobs, smooth=smoothes[cc])
		epsilon_ShearStress[cc] = np.std(ShearStressobs[ind_inter]-ShearStress_smooth[ind_inter])
		
		
		
		Dtau_f[case_study] = ShearStressobs_det[ind_peaks]-ShearStressobs_det[ind_peaks_neg]
		
		#Tprepost[case_study] = (tobs[ind_peaks[1:-1]]-tobs[ind_peaks[:-2]])/(tobs[ind_peaks[2:]]-tobs[ind_peaks[1:-1]])
		Tpre[case_study] = tobs[ind_peaks_center[1:-1]]-tobs[ind_peaks_center[:-2]]
		Tnext[case_study] = tobs[ind_peaks_center[2:]]-tobs[ind_peaks_center[1:-1]]
		Tprepost[case_study] = Tpre[case_study]/Tnext[case_study]
		
		tauf_center[case_study] = ShearStressobs0 + \
			ShearStressobs_det[ind_peaks_center]
		
		id_name.append(case_study)
	
#%%
if flag_load_results==True:
	cc = -1
	for case_study in case_studies:
		dirs['pickles'] = dirs['main']+'/../scenarios/'+case_study+'/pickles/'
		print("")
		print("")
		print(case_study)
		cc+=1
		smoothes = np.zeros((len(case_studies),1))
		if flag_load_LEs==True:
			mhat_tmp, tau_delay_tmp, LEs_tmp, LEs_mean_tmp, LEs_std_tmp, \
				eps_over_L_tmp = load_pickle(\
			    dirs['pickles']+'/LEs_ShearStress_det.pickle', \
			    message='Loading pickled variables: LEs')
			
			LEs_mean[case_study] = LEs_mean_tmp[-1,:]
			LEs_std[case_study] = LEs_std_tmp[-1,:]
#%%

labels = ['ODE', r'SDE$_{yz}$',
		  r'ODE + $\varepsilon_{\tau_f}$',
		  r'SDE$_{yz} + \varepsilon_{\tau_f}$',
		  'Laboratory']
format_color = ['oy', 'sy', 'or', 'sr', 'ok']
fillstyles = ['full', 'full', 'none', 'none', 'none']
legend_order = [4,0,1,2,3]

x_lettera = 0.88
y_lettera = 0.92
x_letterb = 0.88
y_letterb = 0.92
x_letterc = 0.88
y_letterc = 0.92

Ntests = int(len(CV)/Ncases)

plt.figure(figsize=(15,4.5))
gs = gridspec.GridSpec(1, 3, wspace=0.3, hspace=0.0, \
	   top=1.-0.5/(3+1), bottom=0.5/(3+1), left=0.5/(3+1), right=1-0.5/(3+1))

ax1 = plt.subplot(gs[0,0])
plt.grid()
cc = -1
for nn in np.arange(Ntests):
	cc+=1
	CV_cc = []
	for cc in np.arange(Ncases):
		ID = id_name[nn*Ncases + cc]
		CV_tmp = CV[ID]
		CV_cc.append(CV_tmp*100)
	plt.plot(sigman0_nominal, CV_cc, format_color[nn], label=labels[nn], \
		  fillstyle=fillstyles[nn])
handles, labels_leg = plt.gca().get_legend_handles_labels()

plt.legend([handles[idx] for idx in legend_order],
		   [labels_leg[idx] for idx in legend_order],
		   bbox_to_anchor=(2.37, 1.12, 1., .1), ncol=Ntests, \
		   fontsize=16)

plt.xlabel(r'$\sigma_{n0}$ [MPa]')
plt.ylabel(r'CV [$\%$]')
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

ax2 = plt.subplot(gs[0,1])
plt.grid()
cc = -1
for nn in np.arange(Ntests):
	cc+=1
	LEs_mean_max_cc = []
	LEs_std_max_cc = []
	for cc in np.arange(Ncases):
		ID = id_name[nn*Ncases + cc]
		LEs_mean_max_cc.append(LEs_mean[ID][0])
		LEs_std_max_cc.append(LEs_std[ID][0])
	plt.errorbar(sigman0_nominal, LEs_mean_max_cc, yerr=LEs_std_max_cc, \
			  fmt=format_color[nn], markersize=5, label=labels[nn], \
			  fillstyle=fillstyles[nn])
plt.xlabel(r'$\sigma_{n0}$ [MPa]')
plt.ylabel(r'$\lambda_{max}$ [100 Hz]')
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

ax3 = plt.subplot(gs[0,2])
plt.grid()
cc = -1
LTs_mean_max = np.zeros([Ntests,Ncases])
for nn in np.arange(Ntests):
	cc+=1
	LTs_mean_max_cc = []
	LTs_std_max_cc = []
	for cc in np.arange(Ncases):
		ID = id_name[nn*Ncases + cc]
		LTs_mean_max_cc.append(1/LEs_mean[ID][0]/100)
		LTs_std_max_cc.append(LEs_std[ID][0])
		LTs_mean_max[nn,cc] = 1/LEs_mean[ID][0]/100
	plt.errorbar(sigman0_nominal, LTs_mean_max_cc, yerr=LTs_std_max_cc, \
			  fmt=format_color[nn], markersize=5, label=labels[nn], \
			  fillstyle=fillstyles[nn])
plt.xlabel(r'$\sigma_{n0}$ [MPa]')
plt.ylabel(r'$t_{Lyap}$ [s]')
ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

x0,x1 = ax1.get_xlim()
y0,y1 = ax1.get_ylim()
ax1.set_aspect((x1-x0)/(y1-y0))
x0,x1 = ax2.get_xlim()
y0,y1 = ax2.get_ylim()
ax2.set_aspect((x1-x0)/(y1-y0))
x0,x1 = ax3.get_xlim()
y0,y1 = ax3.get_ylim()
ax3.set_aspect((x1-x0)/(y1-y0))

x1lim = ax1.get_xlim()
y1lim = ax1.get_ylim()
ax1.text(x1lim[0]+x_lettera*(x1lim[1]-x1lim[0]), \
		 y1lim[0]+y_lettera*(y1lim[1]-y1lim[0]),'(a)')
x2lim = ax2.get_xlim()
y2lim = ax2.get_ylim()
ax2.text(x2lim[0]+x_letterb*(x2lim[1]-x2lim[0]), \
		 y2lim[0]+y_letterb*(y2lim[1]-y2lim[0]),'(b)')
x3lim = ax3.get_xlim()
y3lim = ax3.get_ylim()
ax3.text(x3lim[0]+x_letterc*(x3lim[1]-x3lim[0]), \
		 y3lim[0]+y_letterc*(y3lim[1]-y3lim[0]),'(c)')
plt.tight_layout()
if flag_save_fig==True:
	plt.savefig(output_file+'.png', dpi=dpi, bbox_inches='tight')

#%%

case_studies_lab = ['labquakes/MeleVeeduetal2020/b724',\
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

diff_eq_type = diff_eq_types[0]
Anoise = Anoises[0]
case_studies_ODE = ['synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b724', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b722', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b696', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b726', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b694', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b697', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b693', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b698', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b695', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b728', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b721', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b725', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b727', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/i417']

diff_eq_type = diff_eq_types[0]
Anoise = Anoises[1]
case_studies_ODE_noise = ['synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b724', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b722', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b696', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b726', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b694', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b697', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b693', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b698', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b695', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b728', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b721', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b725', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b727', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/i417']

diff_eq_type = diff_eq_types[1]
Anoise = Anoises[0]
case_studies_SDE = ['synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b724', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b722', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b696', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b726', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b694', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b697', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b693', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b698', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b695', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b728', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b721', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b725', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b727', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/i417']

diff_eq_type = diff_eq_types[1]
Anoise = Anoises[1]
case_studies_SDE_noise = ['synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b724', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b722', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b696', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b726', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b694', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b697', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b693', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b698', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b695', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b728', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b721', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b725', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/b727', \
				'synthetic/'+diff_eq_type+'_'+str('%.2e' %(Anoise))+'/i417']
	
CV_lab = np.zeros((Ncases,))
for cc in np.arange(Ncases):
	CV_lab[cc] = CV[case_studies_lab[cc]]

CV_ODE = np.zeros((Ncases,))
for cc in np.arange(Ncases):
	CV_ODE[cc] = CV[case_studies_ODE[cc]]

CV_ODE_noise = np.zeros((Ncases,))
for cc in np.arange(Ncases):
	CV_ODE_noise[cc] = CV[case_studies_ODE_noise[cc]]

CV_SDE = np.zeros((Ncases,))
for cc in np.arange(Ncases):
	CV_SDE[cc] = CV[case_studies_SDE[cc]]

CV_SDE_noise = np.zeros((Ncases,))
for cc in np.arange(Ncases):
	CV_SDE_noise[cc] = CV[case_studies_SDE_noise[cc]]

#%%
MAE_ODE = np.mean(np.abs(CV_ODE-CV_lab))
MAE_ODE_noise = np.mean(np.abs(CV_ODE_noise-CV_lab))
MAE_SDE = np.mean(np.abs(CV_SDE-CV_lab))
MAE_SDE_noise = np.mean(np.abs(CV_SDE_noise-CV_lab))
RMSE_ODE = np.sqrt(np.mean((CV_ODE-CV_lab)**2))
RMSE_ODE_noise = np.sqrt(np.mean((CV_ODE_noise-CV_lab)**2))
RMSE_SDE = np.sqrt(np.mean((CV_SDE-CV_lab)**2))
RMSE_SDE_noise = np.sqrt(np.mean((CV_SDE_noise-CV_lab)**2))
print("")
print('MAE ODE         = %.4f; RMSE ODE         = %.4f' %(MAE_ODE, RMSE_ODE))
print('MAE ODE + noise = %.4f; RMSE ODE + noise = %.4f' %(MAE_ODE_noise, RMSE_ODE_noise))
print('MAE SDE         = %.4f; RMSE SDE         = %.4f' %(MAE_SDE, RMSE_SDE))
print('MAE SDE + noise = %.4f; RMSE SDE + noise = %.4f' %(MAE_SDE_noise, RMSE_SDE_noise))

