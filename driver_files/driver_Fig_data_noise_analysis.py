#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 17:13:40 2022

@author: vinco
"""

#%% CLEAR WORKSPACE
from IPython import get_ipython
get_ipython().magic('reset -sf')

#%% IMPORT STANDARD MODULES
import __includes__

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from utils.load_utils import import_data
from utils.param_utils import set_param
import warnings
from scipy import stats
from csaps import csaps

plt.rcParams.update({'font.size': 16})

plt.close('all')
tic_begin = time.time()

dirs = {'main' : os.getcwd(),
		'data' : '/Users/vinco/Work/Data/labquakesde/'
		}
			
#%% SELECT CASE STUDY AND SET UP DIRECTORIES
output_dir = dirs['main'] + \
 '/../scenarios/labquakes/MeleVeeduetal2020/_figures/'

file_name = 'Fig_data_noise_analysis'

output_file = output_dir + file_name

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
flag_save_fig  = False
flag_load_data = True

dpi = 300

#%% SET PARAMETERS
Ncases = len(case_studies)

smoothes = [0.9]*Ncases

p_threshold = 0.05

muobs_mean  = np.zeros([Ncases,])
v0          = np.zeros([Ncases,])
std_v0      = np.zeros([Ncases,])
sigman0     = np.zeros([Ncases,])
std_sigman0 = np.zeros([Ncases,])
W_tauf    = np.zeros([Ncases,])
pW_tauf   = np.zeros([Ncases,])
W_sigman  = np.zeros([Ncases,])
pW_sigman = np.zeros([Ncases,])
print("")
for cc,case_study in enumerate(case_studies):
	
	dirs['pickles'] = dirs['main']+'/../scenarios/'+case_study+'/pickles/'
	
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
		
		n_samples = tobs.shape[0]
		
		if exp_name == 'i417':
			p, C_p = np.polyfit(tobs, ecDispobs*1e-3, deg=1, cov=True)
		else:
			p, C_p = np.polyfit(tobs, ecDispobs*1e3, deg=1, cov=True)
		ecDispobs_det = ecDispobs - (p[0]*tobs + p[1])
		v0[cc] = p[0]
		std_v0[cc] = np.sqrt(C_p[0,0])
		
		p, C_p = np.polyfit(tobs, ShearStressobs, deg=1, cov=True)
		ShearStressobs_det = ShearStressobs - (p[0]*tobs + p[1])
		ShearStressobs0 = p[1]
		
		p, C_p = np.polyfit(tobs, muobs, deg=1, cov=True)
		muobs_det = muobs - (p[0]*tobs + p[1])
		
		del p, C_p
		
		Xobs = np.array([ShearStressobs_det]).T
		Nt,Nx = Xobs.shape
		X = np.zeros((Nt,Nx))
		for nn in np.arange(Nx):
			X[:,nn] = csaps(tobs, Xobs[:,nn], tobs, smooth=smoothes[nn])
		X_norm = np.zeros((Nt,Nx))
		for ii in np.arange(Nx):
			X_norm[:,ii] = (X[:,ii]-np.min(X[:,ii])) / \
				(np.max(X[:,ii])-np.min(X[:,ii]))
		
		sigman0[cc] = np.mean(NormStressobs)
		std_sigman0[cc] = np.std(NormStressobs-sigman0[cc])
		
		ShearStressobs_det_smoothed = csaps(tobs, 
										   ShearStressobs_det, tobs,
										   smooth=smoothes[cc])
		tauf_res = ShearStressobs_det - ShearStressobs_det_smoothed
		
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", \
								category=UserWarning)
			W_tauf[cc], pW_tauf[cc] = stats.shapiro(tauf_res)
			W_sigman[cc], pW_sigman[cc] = stats.shapiro(\
								   NormStressobs-np.mean(NormStressobs))
		muobs_mean[cc] = np.mean(muobs)
		
		if pW_tauf[cc] >= p_threshold and pW_sigman[cc] >= p_threshold:
			fontname="Times New Roman"
			x_pvalue = 0.58
			y_pvalue = 0.78
			
			x_std = 0.58
			y_std = 0.88
			
			plt.figure(figsize=(12,8))
			plt.suptitle(exp_name, fontsize=28)
			ax = plt.subplot(321)
			plt.plot(tobs, NormStressobs,'k',label=r'$\sigma_n$')
			plt.plot(tobs, [np.mean(NormStressobs)]*n_samples,'r', \
			   label=r'$\sigma_{n0}$')
			plt.text(0.015, 0.85, '(a)', fontname=fontname, fontsize=22,
				transform=ax.transAxes)
			plt.xlabel('Time [s]')
			plt.ylabel(r'$\sigma_n$ [MPa]')
			plt.legend(loc='lower left', ncol=2, fontsize=12)
			ax = plt.subplot(322)
			plt.plot(tobs, ShearStressobs_det,'k',label=r'$\tau_f$')
			plt.plot(tobs, ShearStressobs_det_smoothed,'r', \
			   label=r'$\tilde{\tau}_f$')
			plt.text(0.015, 0.85, '(b)', fontname=fontname, fontsize=22,
				transform=ax.transAxes)
			plt.xlabel('Time [s]')
			plt.ylabel(r'$\tau_f$ [MPa]')
			plt.legend(loc='lower left', ncol=2, fontsize=12)
			ax = plt.subplot(323)
			std_sigman = r'std($\sigma_n - \sigma_{n0}$) = %.3f MPa' \
						 %(np.std(NormStressobs-np.mean(NormStressobs)))
			plt.plot(tobs, NormStressobs-np.mean(NormStressobs),'k')
			plt.text(0.015, 0.85, '(c)', fontname=fontname, fontsize=22,
				transform=ax.transAxes)
			plt.xlabel('Time [s]')
			plt.ylabel(r'$\sigma_n - \sigma_{n0}$ [MPa]')
			ax = plt.subplot(324)
			std_tauf_res = r'std($\tau_f - \tilde{\tau}_f$) = %.3f MPa' \
				%(np.std(tauf_res))
			plt.plot(tobs, tauf_res,'k')
			plt.text(0.015, 0.85, '(d)', fontname=fontname, fontsize=22,
				transform=ax.transAxes)
			plt.xlabel('Time [s]')
			plt.ylabel(r'$\tau_f - \tilde{\tau}_f$ [MPa]')
			ax = plt.subplot(325)
			plt.hist(NormStressobs-np.mean(NormStressobs),100)
			p_sigman = 'p-value = %.2f' %(pW_sigman[cc])
			plt.text(0.015, 0.85, '(e)', fontname=fontname, fontsize=22,
				transform=ax.transAxes)
			plt.text(x_std, y_std, std_sigman, transform=ax.transAxes,
				fontsize=12)
			plt.text(x_pvalue, y_pvalue, p_sigman, transform=ax.transAxes,
				fontsize=12)
			plt.xlabel(r'$\sigma_n - \sigma_{n0}$ [MPa]')
			plt.ylabel(r'$\propto \mathrm{pdf}(\sigma_n - \sigma_{n0})$')
			ax = plt.subplot(326)
			plt.hist(tauf_res,100)
			p_tauf_res = 'p-value = %.2f' %(pW_tauf[cc])
			plt.text(0.015, 0.85, '(f)', fontname=fontname, fontsize=22,
				transform=ax.transAxes)
			plt.text(x_std+0.04, y_std, std_tauf_res, transform=ax.transAxes,
				fontsize=12)
			plt.text(x_pvalue+0.04, y_pvalue, p_tauf_res, transform=ax.transAxes,
				fontsize=12)
			plt.xlabel(r'$\tau_f - \tilde{\tau}_f$ [MPa]')
			plt.ylabel(r'$\propto \mathrm{pdf}(\tau_f - \tilde{\tau}_f)$')
			plt.tight_layout()
			if flag_save_fig == True:
				if exp_name == 'b724':
					plt.savefig(output_file+'.png', dpi=dpi)
					plt.savefig(output_file+'.pdf', dpi=dpi)
				dirs['figures'] = dirs['main']+'/../scenarios/' + \
					case_study+'/figures/'
				plt.savefig(dirs['figures']+'/'+exp_name+'_noise_analysis.png',
					dpi=dpi)
				plt.savefig(dirs['figures']+'/'+exp_name+'_noise_analysis.pdf',
					dpi=dpi)
		
			