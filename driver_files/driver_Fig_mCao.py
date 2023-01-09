#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 08:35:45 2021

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
from utils.dynsys_utils import calc_dim_Cao1997
from utils.param_utils import set_param
import numpy as np
from csaps import csaps
import matplotlib
matplotlib.rcParams['font.family'] = 'times'

plt.close('all')
tic_begin = time.time()

dirs = {'main' : os.getcwd(),
		'data' : '/Users/vinco/Work/Data/labquakesde/'
		}
			
#%% SELECT CASE STUDY AND SET UP DIRECTORIES
output_dir = dirs['main'] + \
 '/../scenarios/labquakes/MeleVeeduetal2020/_figures/'

file_name = 'Fig_mCao'

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
flag_save_fig     = False
flag_load_data    = True
flag_load_LEs     = True

dpi = 300

#%% SET PARAMETERS
Ncases = len(case_studies)

# Embedding dimensions to test
m_list = [3,4,5,6,7,8,9,10,11,12,13,14,15,20,50,100] #[2,3,4,5,6,7,8]
Nm = len(m_list)

mmax = 20
E1_thresh = 0.9
E2_thresh = 0.9

smoothes = [1.0]*Ncases

sigman0 = [0.0]*Ncases


#%%
print("")
if flag_load_LEs==True:
	mhat_best = np.zeros((Ncases,1))
	tau_delay_best = np.zeros((Ncases,1))
for cc,case_study in enumerate(case_studies):
	dirs['pickles'] = dirs['main']+'/../scenarios/'+case_study+'/pickles/'

	print(case_study)
	if flag_load_LEs==True:
		mhat_tmp, tau_delay_tmp, LEs_tmp, LEs_mean_tmp, LEs_std_tmp, \
			eps_over_L_tmp = load_pickle(\
		    dirs['pickles']+'/LEs_ShearStress_det.pickle', \
		    message='Loading pickled variables: LEs')
		
		mhat_best[cc] = mhat_tmp
		tau_delay_best[cc] = tau_delay_tmp

#%%
mhat = dict()
E1 = dict()
E2 = dict()
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
		ShearStressobs = data['ShearStress']
		NormStressobs  = data['NormStress']
		
		dtobs = tobs[1]-tobs[0]
		
		p = np.polyfit(tobs, ShearStressobs, deg=1)
		ShearStressobs_det = ShearStressobs - (p[0]*tobs + p[1])
		
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
			
		# CAO (1997)
		mhat[case_study], E1[case_study], E2[case_study] = calc_dim_Cao1997(
					X=X_norm, tau=[tau_delay_best[cc][0]],
					m=np.arange(1,mmax+1,1), E1_thresh=E1_thresh,
					E2_thresh=E2_thresh, qw=None, flag_single_tau=True,
					parallel=False)

#%%
from matplotlib.ticker import MaxNLocator

format_color_legend = ['ok', 'ob', 'or', 'oc', 'og', 'om', 'oy', \
				'sk', 'sb', 'sr', 'sc', 'sg', 'sm', 'sy']
format_color_E1 = ['o-k', 'o-b', 'o-r', 'o-c', 'o-g', 'o-m', 'o-y', \
				's-k', 's-b', 's-r', 's-c', 's-g', 's-m', 's-y']
format_color_E2 = ['o--k', 'o--b', 'o--r', 'o--c', 'o--g', 'o--m', 'o--y', \
				's--k', 's--b', 's--r', 's--c', 's--g', 's--m', 's--y']

figx = 13.5
figy = 6
fontsize = 24
plt.figure(figsize=(figx,figy))
ax = plt.subplot(111)
plt.grid()
for cc,case_study in enumerate(case_studies):
	plt.plot(1,E1[case_study][0,0,0],format_color_legend[cc],
		  label=r'$\sigma_{n0}$ = '+str('%.3f' %(sigman0[cc]))+' MPa',
		  fillstyle='none')
	plt.plot(np.arange(1,mmax,1),E1[case_study][0,0,:],format_color_E1[cc],
		  fillstyle='none')
	plt.plot(np.arange(1,mmax,1),E2[case_study][0,0,:],format_color_E2[cc],
		  fillstyle='none')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend(fontsize=18, ncol=2)
plt.xlabel(r'$m$', fontsize=fontsize)
plt.ylabel(r'$E$', fontsize=fontsize)
plt.tight_layout()
if flag_save_fig == True:
	plt.savefig(output_file+'.png', dpi=dpi, bbox_inches='tight')
	plt.savefig(output_file+'.pdf', dpi=dpi, bbox_inches='tight')