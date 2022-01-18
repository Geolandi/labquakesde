#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 12:52:46 2021

@author: vinco
"""
import pdb

def set_param(case_study):
	if case_study=='labquakes/MeleVeeduetal2020/b693':
		parameters = {
				't0' : 7150.0,
				'tend' : 7350.0,
				'Nheaders' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 350,
				'peaks_height' : 0.6
				}
	elif case_study=='labquakes/MeleVeeduetal2020/b694':
		parameters = {
				't0' : 6330.0,
				'tend' : 6550.0,
				'Nheaders' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 300,
				'peaks_height' : 0.6
				}
	elif case_study=='labquakes/MeleVeeduetal2020/b695':
		parameters = {
				't0' : 7900.0,
				'tend' : 8100.0,
				'Nheaders' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 400,
				'peaks_height' : 0.6
				}
	elif case_study=='labquakes/MeleVeeduetal2020/b696':
		parameters = {
				't0' : 7300.0,
				'tend' : 7500,
				'Nheaders' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 300,
				'peaks_height' : 0.6
				}
	elif case_study=='labquakes/MeleVeeduetal2020/b697':
		parameters = {
				't0' : 9150.0,
				'tend' : 9350.0,
				'Nheaders' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 350,
				'peaks_height' : 0.5
				}
	elif case_study=='labquakes/MeleVeeduetal2020/b698':
		parameters = {
				't0' : 11100.0,
				'tend' : 11300.0,
				'Nheaders' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 350,
				'peaks_height' : 0.6
				}
	elif case_study=='labquakes/MeleVeeduetal2020/b721':
		parameters = {
				't0' : 8500.0,
				'tend' : 8700.0,
				'Nheaders' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 390,
				'peaks_height' : 0.7
				}
	elif case_study=='labquakes/MeleVeeduetal2020/b722':
		parameters = {
				't0' : 7900.0,
				'tend' : 8100.0,
				'Nheaders' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 300,
				'peaks_height' : 0.6
				}
	elif case_study=='labquakes/MeleVeeduetal2020/b724':
		parameters = {
				't0' : 8400.0,
				'tend' : 8600.0,
				'Nheaders' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 300,
				'peaks_height' : 0.6
				}
	elif case_study=='labquakes/MeleVeeduetal2020/b725':
		parameters = {
				't0' : 8550.0,
				'tend' : 8750.0,
				'Nheaders' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 350,
				'peaks_height' : 0.7
				}
	elif case_study=='labquakes/MeleVeeduetal2020/b726':
		parameters = {
				't0' : 9450.0,
				'tend' : 9650.0,
				'Nheaders' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 300,
				'peaks_height' : 0.6
				}
	elif case_study=='labquakes/MeleVeeduetal2020/b727':
		parameters = {
				't0' : 7100.0,
				'tend' : 7300.0,
				'Nheaders' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 390,
				'peaks_height' : 0.7
				}
	elif case_study=='labquakes/MeleVeeduetal2020/b728':
		parameters = {
				't0' : 7677.5,
				'tend' : 7877.5,
				'Nheaders' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 400,
				'peaks_height' : 0.6
				}
	elif case_study=='labquakes/MeleVeeduetal2020/i417':
		parameters = {
				't0' : 3650.0,
				'tend' : 3850.0,
				'Nheaders' : 2,
				'var4peaks' : 'ShearStress',
				'peaks_dist' : 390,
				'peaks_height' : 0.8
				}
	else:
		pdb.set_trace()
	return parameters