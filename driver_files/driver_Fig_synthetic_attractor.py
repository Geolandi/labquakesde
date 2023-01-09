#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:22:11 2021

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
from csaps import csaps
from matplotlib.ticker import MaxNLocator
import matplotlib
from matplotlib import gridspec

matplotlib.rcParams['font.family'] = 'times'
plt.rcParams.update({'font.size': 22})

plt.close('all')
tic_begin = time.time()

dirs = {'main' : os.getcwd(),
		'data' : '/Users/vinco/Work/Data/labquakesde/'
		}
			
#%% SELECT CASE STUDY AND SET UP DIRECTORIES
diff_eq_types = ['param4', 'param6']

Anoise = 0

output_dir = dirs['main'] + \
 '/../scenarios/synthetic/'+diff_eq_types[-1]+'/_figures/'

file_name = 'Fig_attractor'

output_file = output_dir + file_name

#%% SET FLAGS
flag_save_fig     = False
flag_load_data    = True

dpi = 300

#%% SET PARAMETERS

v0 = 10e-6
L = 3e-6
Dt = 200

format_color = ['.-k', '.-b', '.-r', '.-c', '.-g', '.-m', '.-y', \
				'sk', 'sb', 'sr', 'sc', 'sg', 'sm', 'sy']

figx = 16
figy = 7
fontsize = 14

matplotlib.rcParams['font.family'] = 'times'

fig = plt.figure(figsize=(figx,figy))
gs = gridspec.GridSpec(1, 5, wspace=0.3, hspace=0.0, \
	   top=1.-0.5/(3+1), bottom=0.5/(3+1), left=0.5/(3+1), right=1-0.5/(3+1))

ax1 = plt.subplot(gs[0,0])

plt.axis('off')
#plt.text(-0.03, 0.65, 'ODE', rotation=90, fontsize=22)
#plt.text(-0.03, 0.25, r'SDE$_{yu}$', rotation=90, fontsize=22)
plt.text(-0.8, 0.65, 'ODE', rotation=90, fontsize=22)
plt.text(-0.8, 0.25, r'SDE$_{yu}$', rotation=90, fontsize=22)
for dd,diff_eq_type in enumerate(diff_eq_types):
	case_studies = ['synthetic/'+diff_eq_type+'/b726',\
					'synthetic/'+diff_eq_type+'/b698',\
					'synthetic/'+diff_eq_type+'/b728',\
					'synthetic/'+diff_eq_type+'/b721',\
					'synthetic/'+diff_eq_type+'/i417']
	sim_case = ['sim 4', 'sim 8', 'sim 10', 'sim 11', 'sim 14']
	
	Ncases = len(case_studies)
	
	smoothes = [0.9]*Ncases

	sigman0 = [0.0]*Ncases
	
	sigman0 = np.zeros((Ncases,1))
	sigman0_nominal = [0]*len(case_studies)
	epsilon_NormStress = np.zeros((Ncases,1))
	epsilon_ShearStress = np.zeros((Ncases,1))
	
	print("")
	for cc,case_study in enumerate(case_studies):
		
		exp_name = case_study[-4:]
		dirs['pickles'] = dirs['main']+'/../scenarios/'+case_study+'/pickles/'
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
			x = sol_u[sol_t>Tmin,0]
			y = sol_u[sol_t>Tmin,1]
			z = sol_u[sol_t>Tmin,2]
			u = sol_u[sol_t>Tmin,3]
			vobs = v0*np.exp(sol_u[sol_t>Tmin,0])
			ShearStressobs = np.zeros((Nt,))
			ShearStressobs = (tau0 + a*sigman0_cc*sol_u[sol_t>Tmin,1])/1e6
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
			
			ax = fig.add_subplot(2, Ncases, Ncases*dd+cc+1, projection='3d')
			plt.plot(x, y, u, '-k')
			if dd==0:
				plt.title(sim_case[cc], fontsize=22)
			ax.set_xlabel('x', fontsize=fontsize)
			ax.set_ylabel('y', fontsize=fontsize)
			ax.set_zlabel('u', fontsize=fontsize)
			ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))
			ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))
			ax.zaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))
			ax.tick_params(axis = 'both', which = 'major', labelsize=fontsize)

plt.subplots_adjust(left=0.05,
                    bottom=0.05, 
                    right=0.95, 
                    top=0.95, 
                    wspace=0.3, 
                    hspace=-0.4)

#%%
x_letterd = -0.6
y_letterd = 1.0
x1lim = ax1.get_xlim()
y1lim = ax1.get_ylim()
ax1.text(x1lim[0]+x_letterd*(x1lim[1]-x1lim[0]), \
		 y1lim[0]+y_letterd*(y1lim[1]-y1lim[0]),'(d)')
#%%
if flag_save_fig==True:
	plt.savefig(output_file+'.png', dpi=dpi, bbox_inches='tight')