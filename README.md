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
## Citation

If you use this repository, please cite the software release:

```bibtex
@software{talafha_pinn_sft_2026,
  author       = {Talafha, Mohammed H.},
  title        = {PINN--SFT Solar Surface Flux Transport Code},
  year         = {2026},
  version      = {v1.0.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

You may also cite the accompanying manuscript:

```bibtex
@article{talafha2026pinnsft,
  author  = {Talafha, Mohammed H. and Athalathil, J. J.},
  title   = {Data-Constrained Modelling of Solar Surface Magnetic Field Using Physics-Informed Neural Networks},
  journal = {Astronomy \& Astrophysics},
  year    = {2026},
  note    = {submitted}
}
```

The repository citation metadata is also provided in the `CITATION.cff` file.

```
```
