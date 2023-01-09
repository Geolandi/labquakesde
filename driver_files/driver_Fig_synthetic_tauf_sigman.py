#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 13:28:32 2022

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
from matplotlib import gridspec

import matplotlib
matplotlib.rcParams['font.family'] = 'times'

plt.close('all')
tic_begin = time.time()

dirs = {'main' : os.getcwd(),
		'data' : '/Users/vinco/Work/Data/labquakesde/'
		}
			
#%% SELECT CASE STUDY AND SET UP DIRECTORIES
#diff_eq_type = 'param1'
#diff_eq_type = 'param2'
#diff_eq_type = 'param3'
diff_eq_type = 'param4'

output_dir = dirs['main'] + \
 '/../scenarios/synthetic/'+diff_eq_type+'/_figures/'

file_name1 = 'Fig_tauf_ts_param_selection'
file_name2 = 'Fig_tauf_sigman_param_selection'

output_file1 = output_dir + file_name1
output_file2 = output_dir + file_name2

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
sigman0_nominal = [0]*len(case_studies)

figx = 15
figy = 9
fontsize = 24

if diff_eq_type == 'param1':
	title_string = 'ODE\n$\\varepsilon = -0.017 \\times 10^{-3}~~~~~\\nu = \\nu_0~~~~~c_* = 10~s^{-1}$'
elif diff_eq_type == 'param2':
	title_string = 'ODE\n$\\varepsilon = -0.017 \\times 10^{-3}~~~~~\\nu = \\nu_0~~~~~c_* = 1~s^{-1}$'
elif diff_eq_type == 'param3':
	title_string = 'ODE\n$\\varepsilon = -0.017 \\times 10^{-3}~~~~~\\nu = \\nu_0~~~~~c_* = 0.1~s^{-1}$'
elif diff_eq_type == 'param4':
	title_string = 'ODE\n$\\varepsilon = -0.017 \\times 10^{-3}~~~~~\\nu = 20\\nu_0~~~~~c_* = 0.1~s^{-1}$'
	
plt.figure(figsize=(figx, figy))
plt.grid()
ax = plt.subplot(111)
print("")
for cc,case_study in enumerate(case_studies):
	
	exp_name = case_study[-4:]
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
		ShearStressobs = (tau0 + a*sigman0_cc*sol_u[sol_t>Tmin,1])/1e6
		
		dtobs = tobs[1]-tobs[0]
		
		p = np.polyfit(tobs, ShearStressobs, deg=1)
		ShearStressobs_det = ShearStressobs - (p[0]*tobs + p[1])
		ShearStressobs0 = p[1]
		
		del p
		
		Xobs = np.array([ShearStressobs_det]).T
		Nt,Nx = Xobs.shape
		
		plt.plot(tobs, ShearStressobs0 + ShearStressobs_det, '.-k')
		#plt.text(tobs[0]-10, \
		#	0.25+np.mean(ShearStressobs0 + ShearStressobs_det), exp_name, fontsize=18)
		plt.text(tobs[0]-10, 0.05+np.max(ShearStressobs0 + ShearStressobs_det),
			sim_case[cc], fontsize=18)
		
plt.suptitle(title_string, x=0.51, y=0.97, fontsize=fontsize)
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

dirs['figures'] = dirs['main']+'/../scenarios/synthetic/'+diff_eq_type
if flag_save_fig==True:
	plt.savefig(output_file1+'.png', dpi=dpi)
	 
#%%
case_studies = ['synthetic/'+diff_eq_type+'/b726',\
				'synthetic/'+diff_eq_type+'/b698',\
				'synthetic/'+diff_eq_type+'/b728',\
				'synthetic/'+diff_eq_type+'/i417']

figx = 20
figy = 6
fontsize = 18
fontsize_title = 20
matplotlib.rcParams['font.size'] = 12

t0 = 50
t1 = 75
if diff_eq_type == 'param1':
	t0_zoom = 66
	t1_zoom = 68
elif diff_eq_type == 'param2':
	t0_zoom = 69.3
	t1_zoom = 71.3
elif diff_eq_type == 'param3':
	t0_zoom = 55.5
	t1_zoom = 57.5
elif diff_eq_type == 'param4':
	t0_zoom = 62
	t1_zoom = 64

print("")
fig = plt.figure(figsize=(figx, figy))
lr = 0.1
gs = gridspec.GridSpec(2, 5, wspace=0.45, hspace=0.0, \
	   top=1.-0.2/(3+1), bottom=0.8/(3+1), left=0.35/(3+1), right=1-0.05/(3+1))
for cc,case_study in enumerate(case_studies):
	
	exp_name = case_study[-4:]
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
		ShearStressobs = (tau0 + a*sigman0_cc*sol_u[sol_t>Tmin,1])/1e6
		NormStressobs = sigman0_nominal[cc] + \
			(sigman0_nominal[cc]*a/mu0)*sol_u[sol_t>Tmin,3]
		
		dtobs = tobs[1]-tobs[0]
		
		p = np.polyfit(tobs, ShearStressobs, deg=1)
		ShearStressobs_det = ShearStressobs - (p[0]*tobs + p[1])
		ShearStressobs0 = p[1]
		del p
		
		Xobs = np.array([ShearStressobs_det]).T
		Nt,Nx = Xobs.shape
		
		if exp_name=='b726':
			ax1 = plt.subplot(gs[0,0])
			ax1.set_xticklabels([])
			ax1.plot(tobs, ShearStressobs0 + ShearStressobs_det, '.-k')
			#import pdb; pdb.set_trace()
			ax1.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
			plt.xlim([t0,t1])
			plt.title(sim_case[0], fontsize=fontsize_title)
			plt.ylabel(r'$\tau_f$ [MPa]', fontsize=fontsize)
			plt.grid()
			ax2 = plt.subplot(gs[1,0])
			ax2.plot(tobs, NormStressobs, '.-k')
			ax2.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
			plt.xlim([t0,t1])
			plt.grid()
			plt.ylabel(r'$\sigma_{n}$ [MPa]', fontsize=fontsize)
			plt.xlabel('Time [s]', fontsize=fontsize)
		elif exp_name=='b698':
			ax4 = plt.subplot(gs[0,1])
			ax4.set_xticklabels([])
			ax4.plot(tobs, ShearStressobs0 + ShearStressobs_det, '.-k')
			ax4.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
			plt.xlim([t0,t1])
			plt.title(sim_case[1], fontsize=fontsize_title)
			plt.grid()
			ax5 = plt.subplot(gs[1,1])
			ax5.plot(tobs, NormStressobs, '.-k')
			ax5.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
			plt.xlim([t0,t1])
			plt.grid()
			#yticks = [17.35,17.4]
			#ax5.set_yticks(yticks)
			plt.xlabel('Time [s]', fontsize=fontsize)
		elif exp_name=='b728':
			ax6 = plt.subplot(gs[0,2])
			ax6.set_xticklabels([])
			ax6.plot(tobs, ShearStressobs0 + ShearStressobs_det, '.-k')
			ax6.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
			plt.xlim([t0,t1])
			plt.title(sim_case[2], fontsize=fontsize_title)
			plt.grid()
			ax7 = plt.subplot(gs[1,2])
			ax7.plot(tobs, NormStressobs, '.-k')
			ax7.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
			plt.xlim([t0,t1])
			plt.grid()
			#yticks = [17.35,17.4]
			#ax7.set_yticks(yticks)
			plt.xlabel('Time [s]', fontsize=fontsize)
		elif exp_name=='i417':
			ax8 = plt.subplot(gs[0,3])
			ax8.set_xticklabels([])
			ax8.plot(tobs, ShearStressobs0 + ShearStressobs_det, '.-k')
			ax8.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
			plt.xlim([t0,t1])
			[ymin, ymax] = ax8.get_ylim()
			plt.vlines(x=t0_zoom, ymin=ymin, ymax=ymax, color='b',
					  linestyles='dashed')
			plt.vlines(x=t1_zoom, ymin=ymin, ymax=ymax, color='b',
					  linestyles='dashed')
			plt.title(sim_case[4], fontsize=fontsize_title)
			plt.grid()
			ax9 = plt.subplot(gs[1,3])
			ax9.plot(tobs, NormStressobs, '.-k')
			ax9.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
			plt.xlim([t0,t1])
			ylim = ax9.get_ylim()
			dy = ylim[1]-ylim[0]
			plt.vlines(x=t0_zoom, ymin=ylim[0]-dy, ymax=ylim[1]+dy, color='b',
					  linestyles='dashed')
			plt.vlines(x=t1_zoom, ymin=ylim[0]-dy, ymax=ylim[1]+dy, color='b',
					  linestyles='dashed')
			plt.ylim([ylim[0],ylim[1]])
			plt.grid()
			plt.xlabel('Time [s]', fontsize=fontsize)
			
			ax10 = plt.subplot(gs[0,4])
			ax10.set_xticklabels([])
			ax10.plot(tobs, ShearStressobs0 + ShearStressobs_det, '.-k')
			ax10.tick_params(axis='both', which='major', labelsize=fontsize)
			plt.xlim([t0_zoom,t1_zoom])
			[ymin, ymax] = ax7.get_ylim()
			plt.title(sim_case[4], fontsize=fontsize_title)
			plt.grid()
			ax11 = plt.subplot(gs[1,4])
			ax11.plot(tobs, NormStressobs, '.-k')
			ax11.tick_params(axis='both', which='major', labelsize=fontsize)
			plt.grid()
			plt.xlim([t0_zoom,t1_zoom])
			plt.xlabel('Time [s]', fontsize=fontsize)

x_letterb = -0.4
y_letterb = 1.0
x1lim = ax1.get_xlim()
y1lim = ax1.get_ylim()
ax1.text(x1lim[0]+x_letterb*(x1lim[1]-x1lim[0]), \
		y1lim[0]+y_letterb*(y1lim[1]-y1lim[0]),'(b)', fontsize=24)

dirs['figures'] = dirs['main']+'/../scenarios/synthetic/'+diff_eq_type
if flag_save_fig==True:
 	plt.savefig(output_file2+'.png', dpi=dpi)