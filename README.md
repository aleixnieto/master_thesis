# Graph Neural Network-Based Preconditioners for Solving Large-Scale Linear Systems via GMRES

## Notebooks:
### Arnoldi.
- Arnoldi: Arnoldi algorithm full basis.
- Arnoldi_one_iter: One new vector for the orthonormal basis using Arnoldi's algorithm.
-----
### GMRES. There are different versions, each one upgrading the previous one. The first three versions use full arnoldi algorithm whereas the fourth one uses arnoldi_one_iter at each step of the GMRES algorith. This way the fourth one save a lot of time as we only compute one more orthonormal vector at a time instead of the whole orthonormal basis.
- GMRES_BASIC: Raw implementation of GMRES using Arnoldi.
- GMRES_RESTARTED: Added restarting.
- GMRES_PRECONDITIONED: Added preconditioning.
- GMRES_OPTIMIZED: Optimized with Arnoldi_one_iter.
----
### Graph neural networks.
- GNN_toy: First approach to GNNs.
------
------
## Python files:
- functions.py: arnoldi and backsub functions used in GMRES.
- preconditioner.py: functions for preconditioning both the residual and the arnoldi iteration.
