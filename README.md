# SphericalStability
The code is used to numerically investigate the spherical stability of an acoustic cavitation bubble under dual-frequency excitation.
The radial dynamics is described by the Keller--Miksis equation, which is a second order ordinary differential equation. 
The surface dynamics is modelled by a set of linear ordinary differential equation, which takes into account the effect of vorticity by boundary layer approximation.
For numerical calculations, the MPGOS program package is used, which is a general purpose program package written in C++ and CUDA C, and capable to exploit the massive computational power of graphical processing units (GPUs).
