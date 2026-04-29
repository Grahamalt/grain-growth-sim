# Problem Statement

Transparent ceramic scintillators (e.g. Lu2O3:Eu, YAG:Ce, GGAG:Ce)
require both a high concentration of activator dopant for radiative
yield and a microstructure that does not strongly scatter the
emitted light. Activator dopants segregate to grain boundaries
during sintering, retarding grain growth via the solute-drag effect
and pinning the final grain size. Dopant loading therefore couples
two design objectives at once: scintillation yield (more dopant
helps) and optical transparency (smaller grains in the geometric
regime hurt).

This project asks: in a simple Potts-model simulation with explicit
solute segregation and drag, how does the final grain size scale
with dopant concentration, and what does that imply for the
optical-transparency / activator-loading trade-off?

We address it with a 2D Q-state Potts simulation, an Option-A
continuous solute field with diffusion and segregation, a coupled
Metropolis acceptance rule with a (1 - C_local) drag factor, and a
deliberately simple geometric-regime optical-scattering proxy. Full
methods, results, and limitations are in `docs/final_report.md`.
