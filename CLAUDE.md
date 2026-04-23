# CLAUDE.md — grain-growth-sim

## Session startup
1. Read this file and `problem_statement.md`
2. Skim `code/` and `tests/` to orient
3. Summarize current status to the user before coding

## Project overview
Potts-model Monte Carlo simulation of grain growth. Includes:
- Core Potts model (lattice, neighbors, energy)
- Monte Carlo step with Metropolis acceptance
- Solute field with segregation and diffusion
- Grain size extraction and statistics
- Optical scattering proxy from microstructure
- Visualization

## Repository layout
```
code/         simulation modules
tests/        unit tests
results/      figures and data (generated)
docs/         theory and notes
```

## Rules
- Commit after every working increment
- Keep modules small and tested
- Do not commit large binaries to `results/`

## Current status
Scaffold created — modules are stubs. Start with `code/potts.py` and its test.
