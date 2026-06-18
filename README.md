# PINN--SFT Solar Surface Flux Transport Code

This repository contains the code used to reconstruct and forecast the large-scale evolution of the solar surface magnetic field using a physics-informed neural network (PINN) constrained by a source-informed surface flux transport (SFT) model. The framework is designed for cycle-by-cycle reconstruction of Solar Cycles 21--24, validation against WSO synoptic-map data and finite-volume reference solutions, and predictive extension to the ongoing Solar Cycle 25.

The repository includes tools for

* preprocessing and balancing WSO synoptic-map data,
* inferring cycle-dependent effective source terms from the transport equation,
* training PINN solutions for individual solar cycles,
* generating diagnostics such as polar-cap evolution, reversal chronology, dipole evolution, residual maps, and source maps,
* comparing the PINN solutions with finite-volume reference integrations, and
* producing source-informed forecasts for partially observed cycles.

This code accompanies the manuscript on source-informed PINN--SFT reconstruction and prediction of the solar surface magnetic field.
