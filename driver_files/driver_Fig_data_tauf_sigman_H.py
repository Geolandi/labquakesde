#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:33:25 2021

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
from utils.load_utils import import_data
from utils.param_utils import set_param
import numpy as np
from csaps import csaps
from matplotlib import gridspec

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

file_name = 'Fig_tauf_sigman_H'

output_file = output_dir + file_name

case_studies = ['labquakes/MeleVeeduetal2020/b726',\
				'labquakes/MeleVeeduetal2020/b698',\
				'labquakes/MeleVeeduetal2020/b728',\
				'labquakes/MeleVeeduetal2020/i417']
sigman = [15.0, 17.4, 20.0, 25.0]


#%% SET FLAGS
flag_save_fig     = False
flag_load_data    = True

dpi = 300

#%% SET PARAMETERS
cc = -1
Ncases = len(case_studies)

smoothes = [0.9]*Ncases

format_color = ['.-k', '.-b', '.-r', '.-c', '.-g', '.-m', '.-y', \
				'sk', 'sb', 'sr', 'sc', 'sg', 'sm', 'sy']

figx = 20
figy = 9
fontsize = 18
fontsize_title = 20

fig = plt.figure(figsize=(figx, figy))
lr = 0.1
gs = gridspec.GridSpec(3, 5, wspace=0.45, hspace=0.0, \
	   top=1.-0.2/(3+1), bottom=0.8/(3+1), left=0.35/(3+1), right=1-0.05/(3+1))

t0 = 50
t1 = 75
t0_zoom = 71
t1_zoom = 73
print("")
for case_study in case_studies:
	print(case_study)
	cc+=1
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
		
		p = np.polyfit(tobs, LayerThickobs, deg=1)
		LayerThickobs_det = LayerThickobs - (p[0]*tobs + p[1])
		v_LayerThickobs = p[0]
		
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
		
		if exp_name=='b726':
			ax1 = plt.subplot(gs[0,0])
			ax1.set_xticklabels([])
			ax1.plot(tobs, ShearStressobs0 + ShearStressobs_det, '.-k')
			ax1.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
			plt.xlim([t0,t1])
			plt.title(exp_name, fontsize=fontsize_title)
			plt.ylabel(r'$\tau_f$ [MPa]', fontsize=fontsize)
			plt.grid()
			ax2 = plt.subplot(gs[1,0])
			ax2.set_xticklabels([])
			ax2.plot(tobs, NormStressobs, '.-k')
			ax2.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
			plt.grid()
			plt.ylabel(r'$\sigma_{n}$ [MPa]', fontsize=fontsize)
			ax2b = ax2.twinx()
			x = tobs[np.argmin(np.abs(tobs-t0)):np.argmin(np.abs(tobs-t1))]
			y = NormStressobs[\
					 np.argmin(np.abs(tobs-t0)):np.argmin(np.abs(tobs-t1))]
			ax2b.plot(x, np.cumsum(y-np.mean(y)), '-r')
			ax2b.tick_params(axis ='y', labelcolor = 'r', labelsize=fontsize)
			yticks = [-0.05,0.05]
			ax2b.set_yticks(yticks)
			plt.xlim([t0,t1])
			ax3 = plt.subplot(gs[2,0])
			ax3.plot(tobs, LayerThickobs, '.-k')
			ax3.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
			plt.xlim([t0,t1])
			ymin = LayerThickobs[np.argmin(np.abs(tobs-t1))]
			ymax = LayerThickobs[np.argmin(np.abs(tobs-t0))]
			dy = (ymax-ymin)/10
			plt.ylim([ymin-dy, ymax+dy])
			plt.xlabel('Time [s]', fontsize=fontsize)
			plt.ylabel(r'$H$ [mm]', fontsize=fontsize)
			plt.grid()
		elif exp_name=='b698':
			ax4 = plt.subplot(gs[0,1])
			ax4.set_xticklabels([])
			ax4.plot(tobs, ShearStressobs0 + ShearStressobs_det, '.-k')
			ax4.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
			plt.xlim([t0,t1])
			plt.title(exp_name, fontsize=fontsize_title)
			plt.grid()
			ax5 = plt.subplot(gs[1,1])
			ax5.set_xticklabels([])
			ax5.plot(tobs, NormStressobs, '.-k')
			ax5.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
			plt.grid()
			yticks = [17.35,17.4]
			ax5.set_yticks(yticks)
			ax5b = ax5.twinx()
			x = tobs[np.argmin(np.abs(tobs-t0)):np.argmin(np.abs(tobs-t1))]
			y = NormStressobs[\
					 np.argmin(np.abs(tobs-t0)):np.argmin(np.abs(tobs-t1))]
			ax5b.plot(x, np.cumsum(y-np.mean(y)), '-r')
			ax5b.tick_params(axis ='y', labelcolor = 'r', labelsize=fontsize)
			plt.xlim([t0,t1])
			ax6 = plt.subplot(gs[2,1])
			ax6.plot(tobs, LayerThickobs, '.-k')
			ax6.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
			plt.xlim([t0,t1])
			ymin = LayerThickobs[np.argmin(np.abs(tobs-t1))]
			ymax = LayerThickobs[np.argmin(np.abs(tobs-t0))]
			dy = (ymax-ymin)/10
			plt.ylim([ymin-dy, ymax+dy])
			plt.xlabel('Time [s]', fontsize=fontsize)
			plt.grid()
		elif exp_name=='b728':
			ax5 = plt.subplot(gs[0,2])
			ax5.set_xticklabels([])
			ax5.plot(tobs, ShearStressobs0 + ShearStressobs_det, '.-k')
			ax5.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
			plt.xlim([t0,t1])
			plt.title(exp_name, fontsize=fontsize_title)
			plt.grid()
			ax6 = plt.subplot(gs[1,2])
			ax6.set_xticklabels([])
			ax6.plot(tobs, NormStressobs, '.-k')
			ax6.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
			plt.grid()
			yticks = [19.9, 20.0]
			ax6.set_yticks(yticks)
			ax6b = ax6.twinx()
			x = tobs[np.argmin(np.abs(tobs-t0)):np.argmin(np.abs(tobs-t1))]
			y = NormStressobs[\
					 np.argmin(np.abs(tobs-t0)):np.argmin(np.abs(tobs-t1))]
			ax6b.plot(x, np.cumsum(y-np.mean(y)), '-r')
			ax6b.tick_params(axis ='y', labelcolor = 'r', labelsize=fontsize)
			plt.xlim([t0,t1])
			ax7 = plt.subplot(gs[2,2])
			ax7.plot(tobs, LayerThickobs, '.-k')
			ax7.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
			plt.xlim([t0,t1])
			ymin = LayerThickobs[np.argmin(np.abs(tobs-t1))]
			ymax = LayerThickobs[np.argmin(np.abs(tobs-t0))]
			dy = (ymax-ymin)/10
			plt.ylim([ymin-dy, ymax+dy])
			plt.xlabel('Time [s]', fontsize=fontsize)
			plt.grid()
		else:
			ax8 = plt.subplot(gs[0,3])
			ax8.set_xticklabels([])
			ax8.plot(tobs, ShearStressobs0 + ShearStressobs_det, '.-k')
			ax8.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
			plt.xlim([t0,t1])
			ymin = 15.001
			ymax = 16.6
			plt.ylim([ymin, ymax])
			plt.vlines(x=t0_zoom, ymin=ymin, ymax=ymax, color='b',
			  linestyles='dashed')
			plt.vlines(x=t1_zoom, ymin=ymin, ymax=ymax, color='b',
			  linestyles='dashed')
			plt.title(exp_name, fontsize=fontsize_title)
			plt.grid()
			ax9 = plt.subplot(gs[1,3])
			ax9.set_xticklabels([])
			ax9.plot(tobs, NormStressobs, '.-k')
			ax9.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
			ylim = ax9.get_ylim()
			dy = ylim[1]-ylim[0]
			plt.vlines(x=t0_zoom, ymin=ylim[0]-dy, ymax=ylim[1]+dy, color='b',
			  linestyles='dashed')
			plt.vlines(x=t1_zoom, ymin=ylim[0]-dy, ymax=ylim[1]+dy, color='b',
			  linestyles='dashed')
			plt.ylim([ylim[0],ylim[1]])
			plt.grid()
			ax9b = ax9.twinx()
			x = tobs[np.argmin(np.abs(tobs-t0)):np.argmin(np.abs(tobs-t1))]
			y = NormStressobs[\
					 np.argmin(np.abs(tobs-t0)):np.argmin(np.abs(tobs-t1))]
			ax9b.plot(x, np.cumsum(y-np.mean(y)), '-r')
			plt.xlim([t0,t1])
			plt.text(75.5,0.25,\
				r'$\sum_t \left( \sigma_n(t)- \sigma_{n0} \right)$ [MPa]', \
				c='r', fontsize=fontsize, rotation="vertical")
			ax9b.tick_params(axis ='y', labelcolor = 'r', labelsize=fontsize)
			ax10 = plt.subplot(gs[2,3])
			ax10.plot(tobs, LayerThickobs*1e-3, '.-k')
			ax10.tick_params(axis='both', which='major', labelsize=fontsize)
			plt.xlim([t0,t1])
			ymin = LayerThickobs[np.argmin(np.abs(tobs-t1))]*1e-3
			ymax = LayerThickobs[np.argmin(np.abs(tobs-t0))]*1e-3
			dy = (ymax-ymin)/10
			plt.ylim([ymin-dy, ymax+dy])
			ylim= ax10.get_ylim()
			plt.vlines(x=t0_zoom, ymin=ylim[0], ymax=ylim[1], color='b',
			  linestyles='dashed')
			plt.vlines(x=t1_zoom, ymin=ylim[0], ymax=ylim[1], color='b',
			  linestyles='dashed')
			plt.xlabel('Time [s]', fontsize=fontsize)
			plt.grid()
			
			ax11 = plt.subplot(gs[0,4])
			ax11.set_xticklabels([])
			ax11.plot(tobs, ShearStressobs0 + ShearStressobs_det, '.-k')
			ax11.tick_params(axis='both', which='major', labelsize=fontsize)
			plt.xlim([t0_zoom,t1_zoom])
			ymin = 15.001
			ymax = 16.6
			plt.ylim([ymin, ymax])
			plt.title('i417', fontsize=fontsize_title)
			plt.grid()
			ax12 = plt.subplot(gs[1,4])
			ax12.set_xticklabels([])
			ax12.plot(tobs, NormStressobs, '.-k')
			ax12.tick_params(axis='both', which='major', labelsize=fontsize)
			plt.grid()
			plt.xlim([t0_zoom,t1_zoom])
			ax13 = plt.subplot(gs[2,4])
			ax13.plot(tobs, LayerThickobs*1e-3, '.-k')
			ax13.tick_params(axis='both', which='major', labelsize=fontsize)
			plt.xlim([t0_zoom,t1_zoom])
			ymin = LayerThickobs[np.argmin(np.abs(tobs-t1_zoom))]*1e-3
			ymax = LayerThickobs[np.argmin(np.abs(tobs-t0_zoom))]*1e-3
			dy = (ymax-ymin)/10
			plt.ylim([ymin-dy, ymax+dy])
			plt.xlabel('Time [s]', fontsize=fontsize)
			plt.grid()
if flag_save_fig==True:
	plt.savefig(output_file+'.png', dpi=dpi, bbox_inches='tight')
	plt.savefig(output_file+'.pdf', dpi=dpi, bbox_inches='tight')
	