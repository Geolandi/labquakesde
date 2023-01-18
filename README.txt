Adriano Gualandi
Istituto Nazionale di Geofisica e Vulcanologia
adriano.gualandi@ingv.it
18/01/2023


This repository contains the files and results of the paper:

Gualandi, A., D. Faranda, C. Marone, M. Cocco and G. Mengaldo (2023)
"Deterministic and stochastic chaos characterize laboratory earthquakes"
Earth and Planetary Science Letters

The code is written in Python and Julia.
The Python requirements are listed in the requirements.txt file.


PYTHON CODE

You will find Python code in the following folders:

labquaksde/driver_files
labquaksde/code

The driver files to run the calculations are:

driver_laboratory.py
driver_synthetic.py

These scripts run the estimation of the dimension and Lyapunov exponents.
The results are stored in pickle format in the folder:

labquakesde/scenarios/labquakes/MeleVeeduetal2020/*
labquakesde/scenarios/synthetic/param*/*

depending on the selected experiment.

The labquakes experiments are run in a loop, and all results will be produced
and stored in the appropriate directory.
The analysis of the synthetic data will run the case of a specific parameter
set, to be selected at lines 34-37 of the driver file.
You also need to specify at lines 39-40 the observational noise level to add to
the synthetic data.

To generate the synthetic data, see the section relative to the Julia code.

The scenarios already contain the results, and you can reproduce the figure with
the various driver files named driver_Fig_*.


JULIA CODE

The Julia code is in the folder:

labquaksde/driver_files/*.jl

The files are used to generate the synthetic data.
The parameters are those relative to the various tested cases.
In particular:

param1 are the parameters relative to Fig. S2
param2 are the parameters relative to Fig. S3
param3 are the parameters relative to Fig. S4
param4 are the parameters relative to Figs. 4 and S8
param5 are the parameters relative to Fig. 6
param6 are the parameters relative to Figs. 8 and S9
param7 are not used in the manuscript, but correspond to the case of SDE with
the deterministic part being already chaotic for some sigma_n0 values (i.e.,
they are the param5 ODE with stochastic perturbation to make them SDE).


DATA

The data are those from the publication

Mele Veedu, D., Giorgetti, C., Scuderi, M., Barbot, S., Marone, C., &
Collettini, C. (2020)
Bifurcations at the stability transition of earthquake faulting
Geophysical Research Letters, 47, e2020GL087985
https://doi.org/10.1029/2020GL087985

and can are available in Open Science Framework (OSF) at

DOI 10.17605/OSF.IO/9DQH7
