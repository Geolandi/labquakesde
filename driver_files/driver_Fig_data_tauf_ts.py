#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 15:31:00 2022

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
output_dir = dirs['main'] + \
 '/../scenarios/labquakes/MeleVeeduetal2020/_figures/'

file_name = 'Fig_tauf_ts.png'
file_name_zoom = 'Fig_tauf_ts_zoom.png'

output_file = output_dir + file_name
output_file_zoom = output_dir + file_name_zoom

case_studies = ['labquakes/MeleVeeduetal2020/b726',\
				'labquakes/MeleVeeduetal2020/b698',\
				'labquakes/MeleVeeduetal2020/b728',\
				'labquakes/MeleVeeduetal2020/b721',\
				'labquakes/MeleVeeduetal2020/i417']
sigman = [15.0, 17.4, 20.0, 22.0, 25.0]

#%% SET FLAGS
flag_save_fig     = False
flag_load_data    = True

dpi = 300

#%% SET PARAMETERS
Ncases = len(case_studies)

# Smoothing just to refine the search of the max and min
smoothes = [0.9]*Ncases

min_dist_peak = 200

markersize = 15
markersize = 12
markersize_zoom = 20

figx = 15
figy = 9
fontsize = 24

tzoom_min = 50
tzoom_max = 75

title_string = 'Laboratory'

#%%
print("")
plt.figure(figsize=(figx, figy))
plt.grid()
ax = plt.subplot(111)
for cc, case_study in enumerate(case_studies):
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
		
		# peaks
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
				np.argmax(-Xobs[ind_peaks_neg[pp]-twin_neg:ind_peaks_neg[pp] +\
				twin_neg,0])
		
		ind_peaks = np.unique(ind_peaks)
		ind_peaks_neg = np.unique(ind_peaks_neg)
		
		Npeaks = ind_peaks.shape[0]
		ind_peaks_center = np.zeros((Npeaks,))
		ind_peaks_center = ind_peaks_center.astype(int)
		for pp in np.arange(Npeaks):
			ind_peaks_center[pp] = int(ind_peaks[pp] + \
				  np.argmin(np.abs(Xobs[ind_peaks[pp]:ind_peaks_neg[pp]+1,0])))
		
		plt.plot(tobs, ShearStressobs0 + ShearStressobs_det, '.-k')
		plt.plot(tobs[ind_peaks], ShearStressobs0 + \
		   ShearStressobs_det[ind_peaks], '.', c='#7570b3', markersize=markersize)
		plt.plot(tobs[ind_peaks_neg], ShearStressobs0 + \
		   ShearStressobs_det[ind_peaks_neg], '.', c='#1b9e77',
		   markersize=markersize)
		plt.plot(tobs[ind_peaks_center], \
		   ShearStressobs0 + ShearStressobs_det[ind_peaks_center], '.',
		   c='#d95f02', markersize=markersize)
		if exp_name=='b721':
			tobs_b721 = tobs
			tauf_b721 = ShearStressobs0 + ShearStressobs_det
			ind_peaks_b721 = ind_peaks
			ind_peaks_neg_b721 = ind_peaks_neg
			ind_peaks_center_b721 = ind_peaks_center
		plt.text(tobs[0]-9, -0.3+np.max(ShearStressobs0 + ShearStressobs_det),
		   exp_name, fontsize=18)
		plt.text(tobs[0]-10, 0.05+np.max(ShearStressobs0 + ShearStressobs_det),
			r'$\sigma_{n0}=$'+str('%.3f' %(np.mean(NormStressobs)))+'$\pm$'+\
			str('%.3f' %(np.std(NormStressobs-np.mean(NormStressobs))))+\
			' MPa', fontsize=14)
plt.hlines(y=13.2, xmin=tzoom_min, xmax=tzoom_max, color='k', linestyle='--')
plt.hlines(y=14.5, xmin=tzoom_min, xmax=tzoom_max, color='k', linestyle='--')
plt.vlines(x=50, ymin=13.2, ymax=14.5, color='k', linestyle='--')
plt.vlines(x=75, ymin=13.2, ymax=14.5, color='k', linestyle='--')
plt.title(title_string, fontsize=24)
plt.xlabel('Time [s]', fontsize=fontsize)
plt.ylabel(r'$\tau_f$ [MPa]', fontsize=fontsize)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
plt.tight_layout()
if flag_save_fig==True:
	plt.savefig(output_file, dpi=dpi, bbox_inches='tight')

#%%
ii = 9
plt.figure();
plt.grid()
ax = plt.subplot(111)
plt.plot(tobs_b721, tauf_b721, '.-k')
plt.plot(tobs_b721[ind_peaks_b721], tauf_b721[ind_peaks_b721],'.',
		 c='#7570b3', markersize=markersize_zoom)
plt.plot(tobs_b721[ind_peaks_neg_b721], tauf_b721[ind_peaks_neg_b721],'.',
		 c='#1b9e77', markersize=markersize_zoom)
plt.plot(tobs_b721[ind_peaks_center_b721], tauf_b721[ind_peaks_center_b721],
		 '.',c='#d95f02', markersize=markersize_zoom)
plt.annotate(s='', \
	xy=(tobs_b721[ind_peaks_center_b721[ii]], \
		-0.01+np.max(tauf_b721)), \
	xytext=(tobs_b721[ind_peaks_center_b721[ii-1]], \
		-0.01+np.max(tauf_b721)), \
	arrowprops=dict(arrowstyle='<->'))
plt.annotate(s='', \
	xy=(tobs_b721[ind_peaks_center_b721[ii+1]], \
		0.01+np.max(tauf_b721)), \
	xytext=(tobs_b721[ind_peaks_center_b721[ii]], \
		0.01+np.max(tauf_b721)), \
	arrowprops=dict(arrowstyle='<->'))
plt.vlines(x = tobs_b721[ind_peaks_center_b721[ii]], \
  ymin = np.mean(tauf_b721), \
  ymax = 0.1+np.max(tauf_b721),
            color = 'k', linestyle = 'dotted')
plt.vlines(x = tobs_b721[ind_peaks_center_b721[ii-1]], \
  ymin = np.mean(tauf_b721), \
  ymax = 0.1+np.max(tauf_b721),
            color = 'k', linestyle = 'dotted')
plt.vlines(x = tobs_b721[ind_peaks_center_b721[ii+1]], \
  ymin = np.mean(tauf_b721), \
  ymax = 0.1+np.max(tauf_b721),
            color = 'k', linestyle = 'dotted')
plt.text(tobs_b721[ind_peaks_center_b721[ii-1]]+0.5, \
	0.02+np.max(tauf_b721),r'$T_{pre}$', fontsize=18)
plt.text(tobs_b721[ind_peaks_center_b721[ii+1]]-3.2, \
	0.04+np.max(tauf_b721),r'$T_{next}$', fontsize=18)
plt.hlines(y = np.mean(tauf_b721), xmin = tobs_b721[0], \
		   xmax = tobs_b721[-1], color = 'k', linestyle = 'dotted')
plt.text(tobs_b721[ind_peaks_center_b721[ii-1]]-4, \
	np.mean(tauf_b721)-0.1,r'$\langle\tau_f\rangle$', fontsize=18)
plt.xlim([tzoom_min,tzoom_max])
plt.ylim([13.2,14.5])
plt.xlabel('Time [s]', fontsize=fontsize)
plt.ylabel(r'$\tau_f$ [MPa]', fontsize=fontsize)
plt.title('b721', fontsize=24)
#ax.yaxis.set_label_position("right")
#ax.yaxis.tick_right()
ax.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
plt.tight_layout()
if flag_save_fig==True:
	plt.savefig(output_file_zoom, dpi=dpi, bbox_inches='tight')
