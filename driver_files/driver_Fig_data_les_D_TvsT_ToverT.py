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
from matplotlib.patches import Ellipse

plt.close('all')
tic_begin = time.time()

dirs = {'main' : os.getcwd(),
		'data' : '/Users/vinco/Work/Data/labquakesde/'
		}
			
#%% SELECT CASE STUDY AND SET UP DIRECTORIES
output_dir = dirs['main'] + \
 '/../scenarios/labquakes/MeleVeeduetal2020/_figures/'

file_name = 'Fig_les_D_TvsT_ToverT.png'
file_name_zoom = 'Fig_les_D_TvsT_ToverT_zoom.png'

output_file = output_dir + file_name
output_file_zoom = output_dir + file_name_zoom

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

fontsize_title = 16
fontsize_axes = 24 #12
fontsize_legend = 11 #9.5
fontsize = 24
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
tauf_max = dict()
tauf_min = dict()
t_max = dict()
t_min = dict()
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
		Tprepost[case_study] = Tpre[case_study]/Tnext[case_study]
		
		tauf_center[case_study] = ShearStressobs0 + \
			ShearStressobs_det[ind_peaks_center]
		tauf_max[case_study] = ShearStressobs0 + \
			ShearStressobs_det[ind_peaks]
		tauf_min[case_study] = ShearStressobs0 + \
			ShearStressobs_det[ind_peaks_neg]
		t_max[case_study] = tobs[ind_peaks]
		t_min[case_study] = tobs[ind_peaks_neg]

#%%
D1 = np.zeros((Ncases,Nm,Nmin_dist))
if flag_load_results==True:
	if flag_load_LEs==True:
		mhat_best = np.zeros((Ncases,1))
		tau_delay_best = np.zeros((Ncases,1))
		LEs_mean = dict()
		LEs_std  = dict()
		eps_over_L = np.zeros((Ncases,1))
	if flag_load_EVT==True:
		d1 = dict()
		ShearStressobs_det = dict()
		D1_std = np.zeros((Ncases,Nm,Nmin_dist))
		d1_min = np.zeros((Ncases,Nm,Nmin_dist))
		d1_max = np.zeros((Ncases,Nm,Nmin_dist))
	
	for cc,case_study in enumerate(case_studies):
		dirs['pickles'] = dirs['main']+'/../scenarios/'+case_study+'/pickles/'
		print("")
		print("")
		print(case_study)
		smoothes = np.zeros((len(case_studies),1))
		if flag_load_EVT==True:
			data = \
				load_pickle(\
			    dirs['pickles']+'/EVT_ShearStress_det.pickle', \
			    message='Loading pickled variables: EVT')
			tH = data[0]
			d1[case_study] = data[1]
			theta = data[2]
			qs_thresh = data[3]
			p_EVT = data[4]
			q0_thresh_EVT = data[5]
			min_dist = data[6]
			m_list = data[7]
			case_study = data[8]
			ShearStressobs_det[case_study] = data[9]
			smoothes[cc] = data[10]
			for dd,mmin_dist in enumerate(min_dist):
				for kk,mm in enumerate(m_list):
					D1[cc,kk,dd] = np.nanmean(d1[case_study][kk,dd,:])
					D1_std[cc,kk,dd] = np.nanstd(d1[case_study][kk,dd,:])
					d1_min[cc,kk,dd] = np.nanmin(d1[case_study][kk,dd,:])
					d1_max[cc,kk,dd] = np.nanmax(d1[case_study][kk,dd,:])
		if flag_load_LEs==True:
			mhat_tmp, tau_delay_tmp, LEs_tmp, LEs_mean_tmp, LEs_std_tmp, \
				eps_over_L_tmp = load_pickle(\
			    dirs['pickles']+'/LEs_ShearStress_det.pickle', \
			    message='Loading pickled variables: LEs')
			
			mhat_best[cc] = mhat_tmp
			tau_delay_best[cc] = tau_delay_tmp
			LEs_mean[case_study] = LEs_mean_tmp[-1,:]
			LEs_std[case_study] = LEs_std_tmp[-1,:]
			eps_over_L[cc] = eps_over_L_tmp

#%%
nstd = 1

D_KY = np.zeros([Ncases,],dtype=object)
D_KY_min = np.zeros([Ncases,],dtype=object)
D_KY_max = np.zeros([Ncases,],dtype=object)
for cc,case_study in enumerate(case_studies):
	LE_mean = LEs_mean[case_study]
	LE_std  = LEs_std[case_study]
	NLE = LE_mean.shape[0]
	D = np.where(np.cumsum(LE_mean)>0)[0][-1]+1
	D_KY[cc] = D + np.sum(LE_mean[:D])/np.abs(LE_mean[D])
	if np.sum(np.cumsum(LE_mean-nstd*LE_std)>0)==0:
		Dmin = 0
	else:
		Dmin = np.where(np.cumsum(LE_mean-nstd*LE_std)>0)[0][-1]+1
	D_KY_min[cc] = np.min([D_KY[cc],\
				   Dmin + np.sum(LE_mean[:Dmin]-nstd*LE_std[:Dmin]) / \
					np.abs(LE_mean[Dmin])])
					#np.abs(LE_mean[Dmin]+nstd*LE_std[Dmin])
	Dmax = np.where(np.cumsum(LE_mean+nstd*LE_std)>0)[0][-1]+1
	if Dmax<LE_mean.shape[0]:
		D_KY_max[cc] = Dmax + np.sum(LE_mean[:Dmax]+nstd*LE_std[:Dmax]) / \
				np.abs(LE_mean[Dmax])
				#np.abs(LE_mean[Dmax]-nstd*LE_std[Dmax])
	else:
		D_KY_max[cc] = Dmax + np.sum(LE_mean[:Dmax]+nstd*LE_std[:Dmax])
	D_KY_max[cc] = np.max([D_KY[cc], D_KY_max[cc]])

#%%
D1_EVT_diff_threshold = 0.01
D1_best = np.zeros((Ncases,1))
D1_best_min = np.zeros((Ncases,1))
D1_best_max = np.zeros((Ncases,1))
D1_taudelay_max = np.nan*np.zeros((Ncases,1))
D1_best_std = np.zeros((Ncases,1))
D1_std_best = np.zeros((Ncases,1))
d1_min_best = np.zeros((Ncases,1))
d1_max_best = np.zeros((Ncases,1))
ind_m_D1_best = np.zeros((Ncases,), int)
ind_min_dist_D1_best = np.zeros((Ncases,), int)
for cc,case_study in enumerate(case_studies):
	D1_tmp = []
	d1_max_tmp = []
	d1_min_tmp = []
	for dd,mmin_dist in enumerate(min_dist):
		if np.where(np.diff(D1[cc,:,dd])<D1_EVT_diff_threshold)[0].shape[0]>0:
			ind_m_D1_best[cc] = np.where(np.diff(D1[cc,:,dd]) < \
								D1_EVT_diff_threshold)[0][-1]
			ind_min_dist_D1_best[cc] = dd
			D1_tmp.append(D1[cc,ind_m_D1_best[cc],dd])
			d1_max_tmp.append(d1_max[cc,ind_m_D1_best[cc],dd])
			d1_min_tmp.append(d1_min[cc,ind_m_D1_best[cc],dd])
			D1_std_best[cc] = D1_std[cc,ind_m_D1_best[cc],dd]
			d1_min_best[cc] = d1_min[cc,ind_m_D1_best[cc],dd]
			d1_max_best[cc] = d1_max[cc,ind_m_D1_best[cc],dd]
	
	if D1_tmp==[]:
		ind_m_D1_best[cc] = np.argmax(D1[cc,:,dd])
		ind_min_dist_D1_best[cc] = dd
		D1_best[cc] = D1[cc,ind_m_D1_best[cc],dd]
		D1_best_min[cc] = D1_best[cc]
		D1_best_max[cc] = D1_best[cc]
		D1_std_best[cc] = D1_std[cc,ind_m_D1_best[cc],dd]
		d1_min_best[cc] = d1_min[cc,ind_m_D1_best[cc],dd]
		d1_max_best[cc] = d1_max[cc,ind_m_D1_best[cc],dd]
		D1_taudelay_max[cc] = D1_best[cc]
		D1_best_std[cc] = 0
	else:
		D1_best[cc] = np.mean(D1_tmp)
		d1_max_best[cc] = np.max(d1_max_tmp)
		d1_min_best[cc] = np.min(d1_min_tmp)
		D1_best_std[cc] = np.std(D1_tmp)
		D1_best_min[cc] = np.min(D1_tmp)
		D1_best_max[cc] = np.max(D1_tmp)
		D1_taudelay_max[cc] = D1_tmp[-1]

#%%

x_lettera = 0.90
y_lettera = 0.88
x_letterb = 0.02
y_letterb = 0.05
x_letterd = 0.02
y_letterd = 0.88
x_lettere = 0.02
y_lettere = 0.88
NLyap_inset = 2

D_legend_loc = 'upper right'

matplotlib.rcParams['font.family'] = 'times'

#%%
fig = plt.figure(figsize=(15,8))

lw = 4

lr = 0.8
gs = gridspec.GridSpec(2, 2, wspace=0.03, hspace=0.03, \
	   top=1.-0.5/(3+1), bottom=0.5/(3+1), left=lr/(3+1), right=1-lr/(3+1))

plt.suptitle('Laboratory', x=0.5, y=1.0, fontsize=28)

ax1 = plt.subplot(gs[0,0])
plt.xlabel('# Lyapunov exponent', fontsize=fontsize_axes)
plt.ylabel(r'$\lambda$ ['+str(np.round(1/dtobs))[:3]+' Hz]',
		   fontsize=fontsize_axes)
plt.grid()
ax1.yaxis.set_ticks([-1.5,-1.0,-0.5,0.0,0.5])
inset_ax = inset_axes(ax1,
                    width="30%", # width = 30% of parent_bbox
                    height=1., # height : 1 inch
                    loc=3,
					bbox_to_anchor=(0.03, 0.15, 1, 1),
					bbox_transform=ax1.transAxes,
					borderpad=0,)
mark_inset(ax1, inset_ax, loc1=2, loc2=1, fc="none", ec="0.5")
plt.grid()
D_KY = np.zeros([Ncases,],dtype=object)
D_KY_min = np.zeros([Ncases,],dtype=object)
D_KY_max = np.zeros([Ncases,],dtype=object)
ymin = 0
ymax = 0
for cc,case_study in enumerate(case_studies):
	LE_mean = LEs_mean[case_study]
	LE_std  = LEs_std[case_study]
	NLE = LE_mean.shape[0]
	D = np.where(np.cumsum(LE_mean)>0)[0][-1]+1
	D_KY[cc] = D + np.sum(LE_mean[:D])/np.abs(LE_mean[D])
	if np.sum(np.cumsum(LE_mean-nstd*LE_std)>0)==0:
		Dmin = 0
	else:
		Dmin = np.where(np.cumsum(LE_mean-nstd*LE_std)>0)[0][-1]+1
	D_KY_min[cc] = np.min([D_KY[cc],\
				   Dmin + np.sum(LE_mean[:Dmin]-nstd*LE_std[:Dmin]) / \
					np.abs(LE_mean[Dmin])])
	Dmax = np.where(np.cumsum(LE_mean+nstd*LE_std)>0)[0][-1]+1
	if Dmax<LE_mean.shape[0]:
		D_KY_max[cc] = Dmax + np.sum(LE_mean[:Dmax]+nstd*LE_std[:Dmax]) / \
				np.abs(LE_mean[Dmax])
	else:
		D_KY_max[cc] = Dmax + np.sum(LE_mean[:Dmax]+nstd*LE_std[:Dmax])
	D_KY_max[cc] = np.max([D_KY[cc], D_KY_max[cc]])
	ax1.errorbar(np.arange(1,NLE+1),LE_mean,yerr=LE_std,fmt=format_color[cc],\
			  markersize='5',label=r'$\sigma_{n0}$ = ' + \
			  str('%.3f' %(sigman0[cc]))+' MPa',\
			  fillstyle='none')
	inset_ax.errorbar(np.arange(1,NLE+1),LE_mean,yerr=LE_std,
			   fmt=format_color[cc],markersize='5',label=r'$\sigma_{n0}$ = '+\
			   str('%.3f' %(sigman0[cc]))+' MPa',fillstyle='none')
	ymin = np.min([ymin, np.min(LE_mean[NLyap_inset-1]-LE_std[NLyap_inset-1])])
	ymax = np.max([ymax,np.max(LE_mean[0]+LE_std[0])])
ax1.tick_params(axis = 'both', which = 'major', labelsize=fontsize_axes)
ax1.xaxis.set_label_position("top")
ax1.xaxis.tick_top()
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
inset_ax.tick_params(axis = 'both', which = 'major', labelsize=fontsize_axes)
inset_ax.yaxis.set_label_position("right")
inset_ax.yaxis.tick_right()
inset_ax.set_xlim([0,NLyap_inset+0.5])
dy = (ymax-ymin)/5
inset_ax.set_ylim([ymin-dy,ymax+dy])
plt.xticks(ticks=np.arange(NLyap_inset)+1)
plt.xlim([0.5*(1+ax1.get_xlim()[0]), NLyap_inset+0.5])

ax2 = plt.subplot(gs[0,1])
plt.grid()
for mm in np.arange(Nm):
	for dd in np.arange(Nmin_dist):
		if mm==0 and dd==0:
			plt.plot(sigman0, D1[:,mm,dd],'.k',label=r'$D_1$')
		else:
			plt.plot(sigman0, D1[:,mm,dd],'.k')
ax2.errorbar(sigman0, D_KY, \
		 yerr=[list(D_KY[:]-D_KY_min[:]), list(D_KY_max[:]-D_KY[:])], \
		 fmt='.', c='r', capsize=5, label=r'$D_{KY}$')
ymin = np.nanmin([np.nanmin(D1),np.nanmin(D_KY_min[:])])
ymax = np.nanmax([np.nanmax(D1),np.nanmax(D_KY_max[:])])
dy = (ymax-ymin)/10
#plt.ylim(ymin-dy, ymax+dy)
plt.ylim(ymin-dy, 6.5)
ax2.yaxis.set_ticks([2,4,6])
xzoom_min = 13.3
xzoom_max = 14.7
yzoom_min = 1.5
yzoom_max = 6.5
plt.hlines(y=yzoom_min,xmin=xzoom_min,xmax=xzoom_max,color='b',linestyle='--')
plt.vlines(x=xzoom_min,ymin=yzoom_min,ymax=yzoom_max,color='b',linestyle='--')
plt.vlines(x=xzoom_max,ymin=yzoom_min,ymax=yzoom_max,color='b',linestyle='--')
plt.xlabel(r'$\sigma_{n0}$ [MPa]', fontsize=fontsize_axes)
plt.ylabel(r'$D$', fontsize=fontsize_axes)
plt.legend(fontsize=fontsize_legend, loc=D_legend_loc)
ax2.xaxis.set_label_position("top")
ax2.xaxis.tick_top()
ax2.tick_params(axis = 'both', which = 'major', labelsize=fontsize_axes)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")


ax3 = plt.subplot(gs[1,0])
box = ax3.get_position()
box.x0 = box.x0 - 0.088
box.x1 = box.x1
ax3.set_position(box)
plt.grid()
ax3.plot(np.nan, np.nan, label=r'$\sigma_{n0}$ [MPa]',c='w')
for cc,case_study in enumerate(case_studies):
	ax3.plot(Tpre[case_study], Tnext[case_study], format_color[cc], \
		  label=str('%.3f' %(sigman0[cc])), \
			  fillstyle='none')
ax3.legend(bbox_to_anchor=(.46, .91, 1., .12), ncol=1, \
		   fontsize=fontsize_legend)
ax3.tick_params(axis = 'both', which = 'major', labelsize=fontsize_axes)
#ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
ax3.xaxis.set_major_locator(MaxNLocator(prune='both', nbins=5, integer=True))
ax3.yaxis.set_major_locator(MaxNLocator(prune='both', nbins=5, integer=True))
plt.xlabel(r'$T_{pre}$ [s]', fontsize=fontsize_axes)
plt.ylabel(r'$T_{next}$ [s]', fontsize=fontsize_axes)
ax3.set_aspect('equal', adjustable='box')
x0 = 2.5
y0 = x0
x1 = 5.0
y1 = x1
t = np.arange(x0, x1, 0.1)
ax3.plot(t, t, '--', c='orange', lw=lw)
xc = 6.4
yc = xc
delta = 45.0  # degrees
ell = Ellipse((xc, yc), 4.35, 5.5, delta, ls='--', lw=lw, fc='None', ec='orange')
ell.set_clip_box(ax3.bbox)
ax3.add_artist(ell)
x0 = 8.0
y0 = x0
x1 = 10.0
y1 = x1
t = np.arange(x0, x1, 0.1)
ax3.plot(t, t, '--', c='orange', lw=lw)

ax4 = plt.subplot(gs[1,1])
Dx = 0.3
cc = -1
plt.grid()
X = np.zeros(0)
Y = np.zeros(0)
Z = np.zeros(0)
for case_study in case_studies:
	cc+=1
	N = Tprepost[case_study].shape[0]
	dx = Dx/(N+1)
	x = sigman0[cc] + np.arange(0,Dx,dx) - 0.5*Dx
	x = x[:-1]
	X = np.concatenate((X,x))
	Y = np.concatenate((Y,Tprepost[case_study]))
	Z = np.concatenate((Z,Dtau_f[case_study][1:-1]))
	ax4.axvspan(x[0], x[-1], alpha=0.1)

plt.scatter(X, Y, 50, Z, cmap=cm.viridis)
plt.xlabel(r'$\sigma_{n0}$ [MPa]', fontsize=fontsize_axes)
plt.ylabel(r'$T_{pre}$ / $T_{next}$', fontsize=fontsize_axes)
ax4.tick_params(axis = 'both', which = 'major', labelsize=fontsize_axes)
ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
ax4.yaxis.tick_right()
ax4.yaxis.set_label_position("right")
ax4.yaxis.set_ticks([1,1.5,2])
x0 = 13.0
x1 = 16.0
t = np.arange(x0,x1,0.1)
ax4.plot(t, np.ones(t.shape[0]), '--', c='orange', lw=lw)
xc = 19.7      #x-position of the center
yc = 1.0     #y-position of the center
aup = 3.75     #radius on the x-axis
bup = 1.0    #radius on the y-axis
adown = 3.75     #radius on the x-axis
bdown = 0.6    #radius on the y-axis
t = np.linspace(0, np.pi, 100)
ax4.plot(xc + aup*np.cos(t), yc + bup*np.sin(t), '--', c='orange', lw=lw)
t = np.linspace(np.pi, 2*np.pi, 100)
ax4.plot(xc + adown*np.cos(t), yc + bdown*np.sin(t), '--', c='orange', lw=lw)
x0 = 23.5
x1 = 25.7
t = np.arange(x0,x1,0.1)
ax4.plot(t, np.ones(t.shape[0]), '--', c='orange', lw=lw)

cbar_ax = fig.add_axes([0.55, 0.475, 0.1, 0.012])
cbar = plt.colorbar(cax=cbar_ax, orientation="horizontal")
cbar.ax.set_xlabel(r'$\Delta \tau_f$ [MPa]', fontsize=fontsize_axes-10)
cbar.ax.tick_params(axis = 'both', which = 'major', labelsize=fontsize_axes-10)

x2lim = ax2.get_xlim()
x4lim = ax4.get_xlim()
ax2.set_xlim([np.min([x2lim[0],x4lim[0]]), np.max([x2lim[1],x4lim[1]])])
ax4.set_xlim([np.min([x2lim[0],x4lim[0]]), np.max([x2lim[1],x4lim[1]])])

x1lim = ax1.get_xlim()
y1lim = ax1.get_ylim()
ax1.text(x1lim[0]+x_lettera*(x1lim[1]-x1lim[0]), \
		 y1lim[0]+y_lettera*(y1lim[1]-y1lim[0]),'(a)', fontsize=fontsize)
x2lim = ax2.get_xlim()
y2lim = ax2.get_ylim()
ax2.text(x2lim[0]+x_letterb*(x2lim[1]-x2lim[0]), \
		 y2lim[0]+y_letterb*(y2lim[1]-y2lim[0]),'(b)', fontsize=fontsize)
x3lim = ax3.get_xlim()
y3lim = ax3.get_ylim()
ax3.text(x3lim[0]+x_letterd*(x3lim[1]-x3lim[0]), \
		 y3lim[0]+y_letterd*(y3lim[1]-y3lim[0]),'(d)', fontsize=fontsize)
x4lim = ax4.get_xlim()
y4lim = ax4.get_ylim()
ax4.text(x4lim[0]+x_lettere*(x4lim[1]-x4lim[0]), \
		 y4lim[0]+y_lettere*(y4lim[1]-y4lim[0]),'(e)', fontsize=fontsize)

if flag_save_fig==True:
	plt.savefig(output_file, dpi=dpi, bbox_inches='tight')

#%%
fig = plt.figure(figsize=(2,8))
gs = gridspec.GridSpec(2, 1, wspace=0.03, hspace=0.03, \
	   top=1.-0.5/(3+1), bottom=0.5/(3+1), left=lr/(3+1), right=1-lr/(3+1))

plt.suptitle('', x=0.5, y=1.0, fontsize=28)
ax = plt.subplot(gs[0,0])
plt.grid()
for mm in np.arange(Nm):
	for dd in np.arange(Nmin_dist):
		if mm==0 and dd==0:
			plt.plot(sigman0[:3], D1[:3,mm,dd],'.k',label=r'$D_1$')
		else:
			plt.plot(sigman0[:3], D1[:3,mm,dd],'.k')
plt.errorbar(sigman0[:3], D_KY[:3], \
		 yerr=[list(D_KY[:3]-D_KY_min[:3]), list(D_KY_max[:3]-D_KY[:3])], \
		 fmt='.', c='r', capsize=5, label=r'$D_{KY}$')
ymin = np.nanmin([np.nanmin(D1),np.nanmin(D_KY_min[:])])
ymax = np.nanmax([np.nanmax(D1),np.nanmax(D_KY_max[:])])
dy = (ymax-ymin)/10
plt.ylim(ymin-dy, ymax+dy)
plt.xlabel(r'$\sigma_{n0}$ [MPa]', fontsize=fontsize_axes)
plt.legend(fontsize=fontsize_legend, loc=D_legend_loc)
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()
ax.tick_params(axis = 'both', which = 'major', labelsize=fontsize_axes)
ax.xaxis.set_ticks([13.5,14])
#ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.yaxis.set_ticks([2,4,6,8,10,12])

x_letterc = 0.02
y_letterc = 0.05

xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.text(xlim[0]+x_letterc*(xlim[1]-xlim[0]), \
		 ylim[0]+y_letterc*(ylim[1]-ylim[0]),'(c)', fontsize=fontsize)


if flag_save_fig==True:
	plt.savefig(output_file_zoom, dpi=dpi)

#%%
LE_mean_max = np.zeros([Ncases,],dtype=object)
LE_std_max  = np.zeros([Ncases,],dtype=object)
for cc,case_study in enumerate(case_studies):
	LE_mean_max[cc] = LEs_mean[case_study][0]
	LE_std_max[cc]  = LEs_std[case_study][0]

print("")
print("")
print("Maximum Lyapunov exponent")
print("")
for cc in np.arange(Ncases):
    print('$%.5f\pm%.5f$ & ' %(LE_mean_max[cc], LE_std_max[cc]))