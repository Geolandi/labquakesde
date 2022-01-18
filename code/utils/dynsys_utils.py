#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:02:35 2021

@author: vinco
"""

import numpy as np
from tqdm import tqdm
from scipy.stats import genpareto, anderson, anderson_ksamp

def embed_1dim(x,tau=1,m=2):
	# Check inputs shape
	if len(x.shape)!=1:
		if len(x.shape)==2:
			Nx, Ny = x.shape
			if Nx==1 and Ny==1:
				raise ValueError("The input x must be a time series with more than one value.")
			else:
				if Nx==1:
					x = x[0,:]
				else:
					x = x[:,0]
		else:
			raise ValueError("The input x must be 1-dimensional.")
	if not isinstance(tau, int):
		raise ValueError("The input tau must be a scalar integer.")
	if not isinstance(m, int):
		raise ValueError("The input m must be a scalar integer.")
	# Now that x is 1-dim, extract number of epochs
	Nt = x.shape[0]
	# Create Hankel matrix
	H = np.zeros((m,Nt-(m-1)*tau))
	for mm in range(m):
		if mm==m-1:
			H[mm,:] = x[mm*tau:].T
		else:
			H[mm,:] = x[mm*tau:-((m-mm-1)*tau)].T
	return H

def _check_input_shape(X):
	if len(X.shape)==1:
		Nt = X.shape[0]
		if Nt<=1:
			raise ValueError("The input X must have at least more than 1 observation.")
		else:
			X_tmp = X
			X = np.empty((Nt,1))
			X[:,0] = X_tmp.T
			del X_tmp
	elif len(X.shape)==2:
		Nx, Ny = X.shape
		if Nx==1 and Ny==1:
			raise ValueError("The input X must be a time series with more than one value.")
		elif Nx==1 and Ny!=1:
			X = X.T
	else:
		raise ValueError("The input X must be 1 or 2-dimensional.")
	
	return X

def embed(X, tau=[1], m=[2], t=None):
	# X: n-dim array of shape Nt, Nx where Nt is the number of observed
	#    epochs and Nx is the number of observables.
	# tau: list of integers containing the delay time for each of the
	#      observables (same length as Nx).
	# m: list of integers containing the embedding dimension for each of
	#    the observables (same length as Nx).
	
	# Check inputs shape
	if len(X.shape)==1:
		Nt = X.shape[0]
		if Nt<=1:
			raise ValueError("The input X must have at least more than 1 observation.")
		else:
			X_tmp = X
			X = np.empty((Nt,1))
			X[:,0] = X_tmp.T
			del X_tmp
	elif len(X.shape)==2:
		Nx, Ny = X.shape
		if Nx==1 and Ny==1:
			raise ValueError("The input x must be a time series with more than one value.")
		elif Nx==1 and Ny!=1:
			X = X.T
	else:
		raise ValueError("The input x must be 1 or 2-dimensional.")
			
	Nt, Nx = X.shape
	Ntau = len(tau)
	Nm = len(m)
	tauint = [int(tautau) for tautau in tau]
	mint = [int(mm) for mm in m]
	if tauint!=tau:
		raise ValueError("The input tau must be a list of integers.")
	else:
		tau = np.array(tau)
	if mint!=m:
		raise ValueError("The input m must be a list of integers.")
	else:
		m = np.array(m)
	if Ntau!=Nx:
		raise ValueError("The length of tau must equal the number of columns of X.")
	if Nm!=Nx:
		raise ValueError("The length of m must equal the number of columns of X.")
	
	NtH = Nt - np.max((m-1)*tau)
	if t is not None:
		tembed = t[-NtH:]
	else:
		tembed = None
	M = int(np.sum(m))
	H = np.zeros((M,NtH))
	kk = 0
	for ii in np.arange(Nx):
		Hii = embed_1dim(X[:,ii], tau=int(tau[ii]), m=int(m[ii]))
		NmHii, NtHii = Hii.shape
		H[kk:kk+m[ii],:] = Hii[:,NtHii-NtH:]
		kk+=m[ii]
	
	return H.T, tembed
	
#########################
### LYAPUNOV SPECTRUM ###
#########################
from sklearn.neighbors import NearestNeighbors
import warnings

def _calc_tangent_map(X, t_step=1, n_neighbors=20, \
					  eps_over_L0=0.05, eps_over_L_fact=1.2, verbose=False):
	"""
	Parameters
	----------
	X : TYPE
		Input data.
		Format: Nt x m, with Nt number of epochs, m number of time series.
	t_step : TYPE, optional
		DESCRIPTION. The default is 1.
	n_neighbors_min : TYPE, optional
		Minimum number of neighbors to use to calculate the tangent map.
		The default is 0, which uses m for n_neighbors_min.
	n_neighbors_max : TYPE, optional
		DESCRIPTION. The default is 20.
	eps_over_L0 : TYPE, optional
		Starting value for the distance to look up for neighbors, expressed as
		a fraction of the attractor size L. The default is 0.05.
	eps_over_L_fact : TYPE, optional
		Factor to increase the size of the neighborhood if not enough
		neighbors were found. The default is 1.2.

	Returns
	-------
	A : TYPE
		Tangent map approximation.
	eps_over_L : TYPE
		Final value for the distance to look up for neighbors, expressed as
		a fraction of the attractor size L, such that there are at least
		n_neighbors for the calculation of the tangent map at each epoch.
	"""	
	# This function calculates an approximation of the tangent map in the
	# phase space.
	
	eps_over_L = eps_over_L0
	# Find number of epochs Nt and dimension m of X
	Nt, m = X.shape
	# Find horizontal extent of the attractor
	L = np.max(X[:,-1])-np.min(X[:,-1])
	flag_calc_map = True
	while flag_calc_map==True:
		# Set epsilon threshold
		eps = eps_over_L*L
		if verbose==True:
			print("eps_over_L = %f" %(eps_over_L))
		# Find first n_neighbors nearest neighbors to each element X[tt,:]
		# The number of neighbors is n_neighbors + 1 (because the
		# n_neighbors distances are calculated also from the point itself
		# and the distance 0 needs to be esxcluded).
		nbrs = NearestNeighbors(n_neighbors=n_neighbors+1,\
						  algorithm='ball_tree').fit(X[:-t_step,:])
		# Find the distances and the indeces of the nearest neighbors
		distances, indices = nbrs.kneighbors(X[:-t_step,:])
		# Find where the distances of the neighbours are larger than the eps
		# threshold
		ii = np.where(distances>eps)
		#pdb.set_trace()
		if len(ii[0])>0:
			eps_over_L = eps_over_L*eps_over_L_fact
		else:
			flag_calc_map = False
	
	# If n_neighbors_min is lower than the minimum number of points
	# required to estimate the tangent map (i.e., lower than the
	# dimension of X), use as minimum number of neighbors the minimum
	# number necessary to calculate the tangent map
	if n_neighbors<m:
		n_neighbors = m
	# Initialize the tangent map matrix A at each epoch tt (if at time tt
	# only n<n_neighbors neighbors have a distance smaller than eps, then
	# retain only n neighbors).
	A = np.empty((Nt-t_step,m,m))
	# For every time step...
	for tt in np.arange(Nt-t_step):
		# The point under exam is X[tt,:]
		x0_nn = X[tt,:]
		# and it moves in X[tt+t_step,:] after t_step
		x0_nn1 = X[tt+t_step,:]
		# Create the variables containing the neighbors at time
		# tt (xneigh_nn) and their evolution after t_step
		# (xneigh_nn1)
		xneigh_nn = X[indices[tt],:]
		xneigh_nn1 = X[indices[tt]+t_step,:]
		# Calculate the distances of the neighbors from the point
		# under exam (exclude the first element of xneigh_nn because
		# it is equal to x0_nn)
		y = xneigh_nn[1:] - x0_nn
		# Calculate the distances of the neighbors' evolution from
		# the evolution in time of the point under exam (exclude the
		# first element of xneigh_nn1 because it is equal to x0_nn1)
		z = xneigh_nn1[1:] - x0_nn1
		# Calculate the tangent map A at time tt using the 
		# pseudo-inverse of y.T
		A[tt,:,:] = np.dot(z.T,np.linalg.pinv(y.T))
	# Return the tangent map A
	return A, eps_over_L
	
def calc_lyap_spectrum(X, dt=1, t_step=1, n_neighbors=20, eps_over_L0=0.05, \
					   eps_over_L_fact=1.2, sampling=['rand',None], n=1000, \
					   NLEs_statistic=100, method="SS85", verbose=True, \
					   flag_calc_tangent_map=True, A=None):
		# sampling: to decide which points to use for the Lyapunov spectrum
		#           estimation. Options:
		#           ['all', None]: Use all the possible trajectories.
		#           ['begin', int]: Start from the beginning of the time
		#                           series and take a new trajectory after int
		#                           steps.
		#           ['mid', int]: Start from the middle of the time series and
		#                           take a new trajectory after int steps.
		#           ['rand', None]: Start from allowed random times.
		Nt, m = X.shape
		
		if method=="SS85":
			# Find horizontal extent of the attractor
			L = np.max(X[:,-1])-np.min(X[:,-1])
			if flag_calc_tangent_map==True:
				if verbose==True:
					tic = time.time()
					print("")
					print("Calculating tangent map: ",end='')
				
				A, eps_over_L = _calc_tangent_map(X,t_step=t_step,\
									n_neighbors=n_neighbors,\
									eps_over_L0=eps_over_L0,\
									eps_over_L_fact=eps_over_L_fact)
				
				if verbose==True:
					print("eps_over_L = %f   %.2f s" %(eps_over_L, \
													   time.time() - tic))
			else:
				if A==None:
					raise ValueError('Tangent map is missing.')
			
			nbrs = NearestNeighbors(n_neighbors=n_neighbors+1,\
							  algorithm='ball_tree').fit(X)
			distances, indices = nbrs.kneighbors(X)
			
			if sampling[0]=='all':
				if sampling[1]==None:
					ts = np.arange(0,Nt-n*t_step,1)
				else:
					raise ValueError('When sampling[0] is ''all'', sampling[1] must be None')
			elif sampling[0]=='mid':
				#ts = np.arange(int(Nt/2),int(Nt/2) + \
				#			NLEs_statistic*sampling[1],sampling[1])
				ts = np.arange(int(Nt/2),int(Nt/2) + \
							Nt-n*t_step,sampling[1])
			elif sampling[0]=='begin':
				ts = np.arange(0,Nt-n*t_step,sampling[1])
			elif sampling[0]=='rand':
				if sampling[1]==None:
					ts = np.sort(np.array([int(ii) for ii in \
					 np.floor(np.random.rand(NLEs_statistic)*\
													   (Nt-n*t_step))]))
				else:
					raise ValueError('When sampling[0] is ''rand'', sampling[1] must be None')
			else:
				raise ValueError('sampling[0] not valid.')	
			#pdb.set_trace()
			NLEs_statistic = len(ts)
			logR = np.zeros((NLEs_statistic,n,m))
			LEs = np.empty((NLEs_statistic,n,m))
			kk = -1
			if verbose==True:
				print("")
				print("Calculating Lyapunov spectrum")
			LEs_mean = np.zeros((n,m))
			LEs_std  = np.zeros((n,m))
			for t0 in tqdm(ts):
				kk+=1
				ind2follow = indices[t0,:]
				distances_ind2follow = distances[t0,:]
				ind2rm = np.where(ind2follow+(t0+n*t_step)>Nt)[0]
				ind2follow = np.delete(ind2follow,ind2rm)
				distances_ind2follow = np.delete(distances_ind2follow,ind2rm)
				jj2rm = distances_ind2follow>(eps_over_L*L)
				ind2follow = np.delete(ind2follow,jj2rm)
				ind2follow = ind2follow[1:]
				e = np.eye(m)
				
				for nn in np.arange(n):
					ii = t0+nn*t_step
					Aii = A[ii,:,:]
					NA = 0
					NR = 0
					if np.sum(np.abs(Aii))==0.0:
						logR[kk,nn:,:] = np.nan
						LEs[kk,nn:,:] = np.nan
						if NA!=0:
							NA = nn
					else:
						Ae = np.dot(Aii,e)
						Q, R = np.linalg.qr(Ae)
						if nn>0:
							logR[kk,nn,:] = np.log(np.abs(np.diag(R)))
							if np.abs(np.sum(logR[kk,nn,:]))==np.inf:
								LEs[kk,nn,:] = np.nan
								if NR!=0:
									NR = nn
							else:
								LEs[kk,nn,:] = (1/(nn*t_step*dt))*\
											np.sum(logR[kk,1:nn,:],axis=0)
							
						e = Q
					if NA!=0:
						LEs[kk,NA:,:] = np.nan
					if NR!=0:
						LEs[kk,NR:,:] = np.nan
					# Temporary ignore RunTimeWarning when performing
					# np.nanmean and np.nanstd on arrays containing only nan
					with warnings.catch_warnings():
						warnings.simplefilter("ignore", \
											category=RuntimeWarning)
						LEs_mean[nn,:] = \
										np.nanmean(LEs[:,nn,:],axis=0)
						LEs_std[nn,:] = \
										np.nanstd(LEs[:,nn,:],axis=0)
			if verbose==True:
				print("")
				print("LEs: ", end='')
				print(LEs_mean[-1,:])
				print("LEs std: ", end='')
				print(LEs_std[-1,:])
			
		else:
			raise ValueError("Select a valid method")
		return LEs, LEs_mean, LEs_std, eps_over_L

############################
### AUTOCORRELATION TIME ###
############################
def _calc_autocorr_time(X):
	
	_check_input_shape(X)
	
	np.random.rand(42)
	Nt, Nx = X.shape
	
	Nbatches  = int(np.ceil(Nt**(1/3)))
	sizebatch = int(np.ceil(Nt**(2/3)))
	ind_tbatch_start = np.ceil((Nt-sizebatch)*np.random.rand(Nbatches))
	
	autocorr_time = np.zeros((Nx,1))
	for ii in np.arange(Nx):
		x = X[:,ii]
		var_x = np.var(x)
		xbatches = np.zeros((Nbatches,sizebatch));
		for bb in np.arange(Nbatches):
			xbatches[bb,:] = \
		        x[int(ind_tbatch_start[bb]):int(ind_tbatch_start[bb]+sizebatch)];
		mu_xbatches = np.mean(xbatches,axis=1);
		var_muxbatches = np.var(mu_xbatches);
		autocorr_time[ii] = sizebatch*var_muxbatches/var_x;
	return autocorr_time

#####################################
### BEST TIME DELAY FOR EMBEDDING ###
#####################################
from nolitsa.delay import dmi

def calc_tau_delay(X,maxtau=None):
	_check_input_shape(X)
	Nt, Nx = X.shape
	print("")
	print("Calculating best delay time with AMI ", end='')
	print("(Fraser and Swinney, 1986): ", end='')
	if maxtau==None:
		maxtau = Nt
	AMI = np.zeros((maxtau,Nx))
	for ii in np.arange(Nx):
		AMI[:,ii] = dmi(X[:,ii],maxtau=maxtau)
	tau_delay = []
	for ii in np.arange(Nx):
		tau_delay.append([i for i,val in enumerate(np.diff(AMI[:,ii])>0) \
														   if val][0])
	print(tau_delay)
	return tau_delay

################################
### BEST EMBEDDING DIMENSION ###
################################
from nolitsa.dimension import afn

def calc_dim_Cao1997(X, tau=np.arange(1,34,4), m=np.arange(1,21,1), \
						E1_thresh=0.95, E2_thresh=0.95, mw=2, qw=3, \
						window=10, flag_single_tau=False, parallel=True):
		if isinstance(tau, int):
			t = tau
			tau = np.empty((1,),dtype=int)
			tau[0] = int(t)
			Ntau=1
		else:
			tau = np.array([int(tautau) for tautau in tau])
			Ntau = tau.shape[0]
		Nobs = X.shape[1]
		if flag_single_tau==True:
			# Check dimension
			if Nobs!=Ntau:
				raise ValueError("When flag_single_tau is True then X.shape[1] must be equal to len(tau).")
		Nm   = m.shape[0]
		if flag_single_tau==False:
			E  = np.empty((Nobs,Ntau,Nm))
			Es = np.empty((Nobs,Ntau,Nm))
			E1 = np.empty((Nobs,Ntau,Nm-1))
			E2 = np.empty((Nobs,Ntau,Nm-1))
			mhat = np.zeros((Nobs,Ntau))
		else:
			E  = np.empty((Nobs,1,Nm))
			Es = np.empty((Nobs,1,Nm))
			E1 = np.empty((Nobs,1,Nm-1))
			E2 = np.empty((Nobs,1,Nm-1))
			mhat = np.zeros((Nobs,1))
		if qw==None:
			mw = 0
			qw = _calc_autocorr_time(X)
		tic = time.time()
		print("")
		print("Calculating best embedding dimension with Cao (1997) method")
		for ii in range(Nobs):
			if flag_single_tau==False:
				tt=-1
				for tautau in tau:
					print("Scalar time series %d/%d, tau = %d: " %(ii+1,Nobs,tautau),end='')
					tt+=1
					#flag_afn = False
					#while flag_afn==False:
					#	try:
					#		E[ii,tt,:],Es[ii,tt,:] = afn(X[:,ii], \
					#				dim=m, tau=tautau, maxnum=mw*window+qw)
					#		flag_afn = True
					#	except:
					#		mw+=30
					#		print("afn failed.")
					#		print("Increasing window to %d" %(mw))
					try:
						E[ii,tt,:],Es[ii,tt,:] = afn(X[:,ii], \
									dim=m, tau=tautau, maxnum=None, \
									window=int(qw[ii]), parallel=parallel)
						E1[ii,tt,:] = E[ii,tt,1:]/E[ii,tt,:-1]
						E2[ii,tt,:] = Es[ii,tt,1:]/Es[ii,tt,:-1]
						if np.sum(E2[ii,tt,:]<E2_thresh)>0:
							indE1 = np.argmax(E1[ii,tt,:]>=E1_thresh)
							mhat[ii,tt] = int(m[indE1])
							print("m = %d; Calculation time: %.2f s" \
									%(mhat[ii,tt], time.time() - tic))
						else:
							print("E2 ~ const.: stochastic time series ", end='')
							print("(m = nan); Calculation time: %.2f s" \
										%(time.time() - tic))
							mhat[ii,tt] = np.nan
					except:
						print("afn failed.")
						mhat[ii,tt] = np.nan
					
			else:
				tautau = tau[ii]
				print("Scalar time series %d/%d, tau = %d: " %(ii+1,Nobs,tautau),end='')
				tt = 0
				#flag_afn = False
				#while flag_afn==False:
				#	try:
				#		E[ii,tt,:],Es[ii,tt,:] = afn(X[:,ii], dim=m, tau=tautau, maxnum=mw*window+qw)
				#		flag_afn = True
				#	except:
				#		mw+=30
				#		print("afn failed.")
				#		print("Increasing window to %d" %(mw))
				try:
					E[ii,tt,:],Es[ii,tt,:] = afn(X[:,ii], dim=m, tau=tautau, \
								  maxnum=None, window=int(qw), parallel=parallel)
					E1[ii,tt,:] = E[ii,tt,1:]/E[ii,tt,:-1]
					E2[ii,tt,:] = Es[ii,tt,1:]/Es[ii,tt,:-1]
					if np.sum(E2[ii,tt,:]<E2_thresh)>0:
						indE1 = np.argmax(E1[ii,tt,:]>=E1_thresh)
						mhat[ii,tt] = int(m[indE1])
						print("m = %d; Calculation time: %.2f s" %(mhat[ii,tt], time.time() - tic))
					else:
						print("E2 ~ const.: stochastic time series (m = nan); Calculation time: %.2f s" %(time.time() - tic))
						mhat[ii,tt] = np.nan
				except:
					print("afn failed.")
					mhat[ii,tt] = np.nan
				
				
		return mhat, E1, E2
	
############################
### EXTREME VALUE THEORY ###
############################
import pdb
import time
import matplotlib.pyplot as plt
from fitter import Fitter
from scipy.stats import probplot
from scipy.stats.mstats import mquantiles


def calc_dtheta_EVT(X, p=2, q0_thresh=0.98, dq_thresh=None, fit_distr=False, \
					verbose=True):
	"""
	Parameters
	----------
	X : numpy array
		2-dim array with shape Nt x Nx, with Nt number of epochs and Nx number
		of time series.
	p : int, optional
		Norm to calculate the distance. The default is 2.
	q0_thresh : float, optional
		Quantile threshold to find the extremes. The default is 0.98.
	dq_thresh : float or None, optional
		If None, the calculations are performed with fixed q0_thresh for every
		epoch.
		If float, the calculations are performed for a given epoch starting
		with threshold equal to q0_thresh. A fit to the distribution of the
		extremes is performed with an exponential and a Generalized Pareto (GP)
		distribution. An Akaike Information Criterion (AIC) is used to assess
		if an exponential distribution is preferred or not. If a GP is
		preferred, reduce the quantile threshold by dq_thresh and redo the
		calculations until an exponential is preferred. The idea behind this is
		that the extremes should follow an exponential in theory, which is a
		particular case of the GP. If a GP fits better the extremes, then the
		extremes are not really representative of the dynamics.
		The default is None.
	verbose : boolean, optional
		If True print information to screen. The default is True.

	Returns
	-------
	d_EVT : numpy array
		Instantaneous (or local) dimension of shape Nt x 1.
	theta_EVT : numpy array
		Instantaneous (or local) extremal index of shape Nt x 1.
	qs_thresh : numpy array
		Final qs_thresh adopted at each time step. Shape Nt x 1.

	"""
	if verbose==True:
		print("Calculating EVT: ")
	#np.random.seed(42)
	Nt,Nts = X.shape
	d_EVT     = np.zeros([Nt,1])
	theta_EVT = np.zeros([Nt,1])
	qs_thresh = np.zeros([Nt,1])
	# Temporary ignore the "RuntimeWarning: divide by zero
	# encountered in log"
	with np.errstate(divide='ignore', invalid='ignore'):
			time.sleep(0.5)
			for tt in tqdm(np.arange(Nt)):
				q_thresh = q0_thresh
				
				delta = np.sum(np.abs(X[:,:] - X[tt,:])**p,axis=1)**(1/p)
				
				g = -np.log(delta)
				
				if dq_thresh==None:
					q = mquantiles(g,q_thresh,alphap=0.5, betap=0.5)
						
					idx_g_over_q = np.argwhere(g > q)
					
					g_over_q = np.sort(g[g>q])
					g_over_q = g_over_q[np.isfinite(g_over_q)]
					
					d_EVT[tt] = 1 / np.mean(g_over_q - q)
							
					T_tt = np.diff(idx_g_over_q,axis=0)
					S_tt = T_tt - 1
					N_c = len(S_tt[S_tt>0])
					N   = len(T_tt)
					qs = 1 - q_thresh
					theta_EVT[tt] = (np.sum(qs * S_tt) + N + N_c - \
					  np.sqrt((np.sum(qs * S_tt) + N + N_c)**2 - \
			 		   8 * N_c * np.sum(qs * S_tt))) / (2 * np.sum(qs * S_tt))
					if fit_distr==True:
						f = Fitter(g_over_q, distributions=['expon','genpareto'])
						f.fit()
						if f._aic['expon']<f._aic['genpareto']:
							qs_thresh[tt] = True
						else:
							qs_thresh[tt] = False
					else:
						qs_thresh[tt] = q_thresh
				else:
					flag_q = False
					while flag_q==False:
						
						q = mquantiles(g,q_thresh,alphap=0.5, betap=0.5)
						
						idx_g_over_q = np.argwhere(g > q)
						
						g_over_q = np.sort(g[g>q])
						g_over_q = g_over_q[np.isfinite(g_over_q)]
					
						f = Fitter(g_over_q, distributions=['expon','genpareto'])
						f.fit()
						#f.summary()
		
						#pdb.set_trace()
						if f._aic['expon']<f._aic['genpareto']:
							flag_q = True
							d_EVT[tt] = 1 / np.mean(g_over_q - q)
							
							T_tt = np.diff(idx_g_over_q,axis=0)
							S_tt = T_tt - 1
							N_c = len(S_tt[S_tt>0])
							N   = len(T_tt)
							qs = 1 - q_thresh
							theta_EVT[tt] = (np.sum(qs * S_tt) + N + N_c - \
							  np.sqrt((np.sum(qs * S_tt) + N + N_c)**2 - \
					 		   8 * N_c * np.sum(qs * S_tt))) / (2 * np.sum(qs * S_tt))
							qs_thresh[tt] = q_thresh
						else:
							q_thresh = q_thresh - dq_thresh
				
				

	return d_EVT, theta_EVT, qs_thresh




# def calc_dtheta_EVT(X, p=2, qs_thresh=[0.98],  alpha=0.01, r_thresh=0.95, r_thresh_perc=0.05):
# 	np.random.seed(42)
# 	Nt,Nts = X.shape
# 	d_EVT = np.zeros([Nt,1])
# 	theta_EVT = np.zeros([Nt,1])
# 	s = np.ones([Nt,1])
# 	r = np.ones([Nt,1])
# 	#Nq = len(qs_thresh)
# 	flag_GP = False
# 	cc = -1
# 	q_thresh = qs_thresh[0]
# 	# Temporary ignore the "RuntimeWarning: divide by zero
# 	# encountered in log"
# 	with np.errstate(divide='ignore', invalid='ignore'):
# # 		while flag_GP==False:
# # 			cc+=1
# # 			try:
# # 				q_thresh = qs_thresh[cc]
# # 			except:
# # 				q_thresh+=(1-q_thresh)/2
# # 			print('')
# # 			print('q_thresh = %f' %(q_thresh))
# # 			time.sleep(0.5)
# 			for tt in tqdm(np.arange(Nt)):
# 				#delta = np.linalg.norm(X[:,:] - X[tt,:], ord=p, axis=1)
# 				delta = np.sum(np.abs(X[:,:] - X[tt,:])**p,axis=1)**(1/p)
# 				
# 				g = -np.log(delta)
# 				#q = np.quantile(g,q_thresh,interpolation='linear')
# 				q = mquantiles(g,q_thresh,alphap=0.5, betap=0.5)
# 				
# 				idx_g_over_q = np.argwhere(g > q)
# 				T_tt = np.diff(idx_g_over_q,axis=0)
# 				S_tt = T_tt - 1
# 				N_c = len(S_tt[S_tt>0])
# 				N   = len(T_tt)
# 				
# 				g_over_q = np.sort(g[g>q])
# 				#g_over_q = g_over_q[0:-1]
# 				g_over_q = g_over_q[np.isfinite(g_over_q)]
# 				#g_over_q = g_over_q[0:-np.sum(g_over_q==np.inf)]
# 				d_EVT[tt] = 1 / np.mean(g_over_q - q)
# 				
# 				# Più alto q possibile. Stima exp e Pareto di d_EVT.
# 				# Confronta intervalli confidenza delle due stime
# 				# Tieni massimo q per cui le due stime sono consistenti
# 				f = Fitter(g_over_q, distributions=['expon','genpareto'])
# 				f.fit()
# 				f.summary()
# # 				aa = probplot(g_over_q, \
# # 						  dist='expon',sparams=f.fitted_param['expon'],\
# # 						  plot=plt,fit=True,rvalue=True)
# # 				plt.close('all')
# # 				r[tt] = aa[1][2]
# 				pdb.set_trace()
# # 				_,_,s[tt] = anderson_ksamp([g_over_q,\
# # 								genpareto.rvs(c=0,scale = 1/d_EVT[tt], \
# # 								  size=g_over_q.shape[0])],midrank=False)
# 	# 			A,B,S = anderson(g_over_q, dist='expon')
# 	# 			if A>B[4]:
# 	# 				s[tt] = 1
# # 				if s[tt]<alpha:
# #  					print('Null hypothesis: extremes follow a GP. Rejected at %f confidence level.' %(alpha))
# #  					time.sleep(0.5)
# #  					break
# # 				if r[tt]<r_thresh:
# # 					pdb.set_trace()
# # 					print('r^2 goodness of fit on Q-Q plot less than %f' %(r_thresh))
# # 					time.sleep(0.5)
# # 					break
# 				
# 				qs = 1 - q_thresh
# 				theta_EVT[tt] = (np.sum(qs * S_tt) + N + N_c - \
# 				  np.sqrt((np.sum(qs * S_tt) + N + N_c)**2 - \
# 		 		   8 * N_c * np.sum(qs * S_tt))) / (2 * np.sum(qs * S_tt))
# 			#pdb.set_trace()
# # 			if np.sum(s<alpha)>0:
# # 				flag_GP=False
# # 			if np.sum(r<r_thresh)/Nt>r_thresh_perc:
# # 				pdb.set_trace()
# # 				flag_GP=False
# # 			else:
# # 				pdb.set_trace()
# # 				flag_GP=True
# 	return d_EVT, theta_EVT, r, q_thresh




#     logdista=-log(pdist2(x(j,:),x));
#     thresh=quantile(logdista, quanti);
# 	logextr=logdista(logdista>thresh);
#     logextr=logextr(isfinite(logextr));
# 	D1(j,1)=1./mean(logextr-thresh);

# def calc_H_EVT(self):
# thetamin = np.min(self.theta_EVT)
# thetamax = np.max(self.theta_EVT)
# thetamean = np.mean(self.theta_EVT)
# if thetamax<1:
# 	Hmax = -np.log(1-thetamax)
# else:
# 	Hmax = np.inf
# Hmean = -np.log(1-thetamean)
# Hmin = -np.log(1-thetamin)
# self.H_EVT = [Hmin, Hmean, Hmax]

# return self

# def calc_tstar_EVT(self):
# self.tstar_EVT = \
# 	[1/self.H_EVT[2], 1/self.H_EVT[1], 1/self.H_EVT[0]]
# return self

# def calc_EVT(self, X, p=2, q_thresh=0.98, verbose=False):
# if verbose==True:
# 	tic = time.time()
# 	print(" ")
# 	print("Characterizing time series via EVT:")
# self = self.calc_dtheta_EVT(X, p=p, q_thresh=q_thresh, verbose=verbose)
# self = self.calc_H_EVT()
# self = self.calc_tstar_EVT()
# if verbose==True:
# 	print("%.2f s" %(time.time()-tic))
# return self

# def calc_extremalindex_Suaveges(self,psi,q_thresh=0.98):
# psi = self._check_input_shape(psi)
# if psi.shape[1]>1:
# 	raise ValueError("The negative log-distances should be a 1-dim array.")
# u = np.quantile(psi[:,0],q_thresh,interpolation='linear')
# q = 1-q_thresh
# Li = np.argwhere(psi[:,0]>u)
# Ti = np.diff(Li, axis=0)
# Si = Ti-1
# Nc = len(Si[Si>0])
# N = len(Ti)
# theta = ( np.sum(q*Si) + N + Nc - \
#    np.sqrt( (np.sum(q*Si) + N + Nc)**2 - 8*Nc*np.sum(q*Si)) ) / \
#    (2*np.sum(q*Si))
# return theta

# def calc_dtheta_EVT_blockmaxima(self,X,p=2,q_thresh=0.98,Ndiv=2,Lb=10,multivariate=True,verbose=False):
# X = self._check_input_shape(X)
# Nt,Nts = X.shape
# if Nts==1:
# 	if verbose==True:
# 		print("The dataset is univariate.")
# 		print("   Setting multivariate to False.")
# 		print("   The distance will be calculate in L1 norm.")
# 	multivariate = False

# Nt_pairs = int(np.floor(Nt/Ndiv))
# #Nt_pairs = Ndiv*Nt_cut
# if Nt_pairs<Lb:
# 	raise ValueError("The trajectories length is too short given the specified blocks length Lb.")
# Nb = int(np.floor(Nt_pairs/Lb))
# Nt_pairs = Nb*Lb
# pairs = list(combinations(range(Ndiv), 2))
# l = len(pairs)
# if verbose==True:
# 	print(" ")
# 	print("Total number of trajectory pairs: %d" %(l))
# 	print("Total number of blocks per trajectory pair: %d" %(Nb))
# # Use all the time series at once
# if multivariate==True:
# 	self.D2_GEV = np.zeros((1,l))
# 	self.DEI = np.zeros((1,l))
# 	x = np.zeros((Nt_pairs,Nts,2))
# 	for ll in np.arange(l):
# 		ind1 = pairs[ll][0]
# 		ind2 = pairs[ll][1]
# 		# Pair of trajectories for all observables
# 		x[:,:,0] = X[ind1*Nt_pairs:(ind1+1)*Nt_pairs,:]
# 		x[:,:,1] = X[ind2*Nt_pairs:(ind2+1)*Nt_pairs,:]
# 		psi = -np.log(np.linalg.norm(x[:,:,0]-x[:,:,1],ord=p,axis=1))
# 		#pdb.set_trace()
# 		logextr = np.ones([Nb,1])
# 		for bb in np.arange(Nb):
# 			logextr[bb] = np.max(psi[bb*Lb:(bb+1)*Lb])
# 		sh, loc, sc = gev.fit(logextr)
# 		self.D2_GEV[0,ll] = 1/sc
# 		#pdb.set_trace()
# 		self.DEI[0,ll] = self.calc_extremalindex_Suaveges(psi=psi,q_thresh=q_thresh)
# # Perform the analysis for each time series, which is treated as a
# # different observable
# elif multivariate==False:
# 	self.D2_GEV = np.zeros((Nts,l))
# 	self.DEI = np.zeros((Nts,l))
# 	for ii in np.arange(Nts):
# 		# For each pair:
# 		x = np.zeros((Nt_pairs,2))
# 		for ll in np.arange(l):
# 			ind1 = pairs[ll][0]
# 			ind2 = pairs[ll][1]
# 			# Pair of trajectories for observable ii
# 			x[:,0] = X[ind1*Nt_pairs:(ind1+1)*Nt_pairs,ii]
# 			x[:,1] = X[ind2*Nt_pairs:(ind2+1)*Nt_pairs,ii]
# 			psi = -np.log(np.abs(np.diff(x)))[:,0]
# 			logextr = np.ones([Nb,1])
# 			for bb in np.arange(Nb):
# 				logextr[bb] = np.max(psi[bb*Lb:(bb+1)*Lb])
# 			sh, loc, sc = gev.fit(logextr)
# 			self.D2_GEV[ii,ll] = 1/sc
# 			#pdb.set_trace()
# 			self.DEI[ii,ll] = self.calc_extremalindex_Suaveges(psi=psi,q_thresh=q_thresh)
# return self