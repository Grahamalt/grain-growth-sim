# Site outline

Working outline for the project web page. Sections are listed in
display order. "Asset" refers to a file that already exists in the
repo (path relative to repo root). MathJax is used for any equation.

---

## 1. Hero / title

- **Title:** *Solute drag in grain growth*
- **Subtitle:** *A 2D Potts-model simulation with an optical
  scattering proxy for transparent ceramic scintillators*
- **Byline:** Graham Altschuler — 3.21 simulation project, Spring 2026
- **Hero visual:** showcase figure
  (`results/figures/showcase_figure.png`).
  - Stretch goal: an animated GIF assembled from
    `results/figures/snapshots/snapshot_*.png`
    (MCS 0 -> 20 -> 100 -> 200) cycled with a 1.5 s frame delay.
- **One-line description (under hero):** "Why does adding more dopant
  to a transparent ceramic scintillator make it less transparent? A
  numerical experiment."

## 2. TL;DR

- 2-3 sentences, plain language.
- Draft: "I simulated grain growth with solute drag in a 2D Q-state
  Potts Monte Carlo, with an explicit segregating-impurity field and
  a simple optical-scattering proxy. Increasing the dopant
  concentration slows boundary motion and shrinks the final grain
  size, which doubles the simulated optical attenuation across the
  range studied — a quantitative illustration of the
  activator-loading vs transparency trade-off in scintillator
  ceramic design."

## 3. Motivation

- Source: `problem_statement.md` (verbatim or near-verbatim) plus the
  background paragraph from `docs/final_report.md` Section 1.
- 3-4 short paragraphs:
  1. What a scintillator does and what "transparent ceramic"
     means.
  2. Why grain size matters: scattering regimes (Rayleigh vs
     geometric) and where typical sintered grain sizes fall.
  3. Why dopants couple to grain size: solute segregation to
     boundaries, drag.
  4. The trade-off framed as a question.
- No figure required.

## 4. Theory

- Source: `docs/theory.md`, sections 1-4 and 6-8.
- Equations rendered via MathJax (already authored in `$$...$$`
  blocks in theory.md).
- Subsections:
  - 4.1 Potts model (Hamiltonian, neighbor stencil, boundary site)
  - 4.2 Metropolis Monte Carlo (proposal + acceptance rules)
  - 4.3 Solute field: diffusion (FD Laplacian + CFL guard)
  - 4.4 Boltzmann segregation isotherm and the mass-conserving
        relaxation step
  - 4.5 Coupling: the (1 - C_local) drag factor in the acceptance
        probability
  - 4.6 Optical scattering proxy (boundary map -> Gaussian halo ->
        mu_eff)
- Symbol table from theory.md Section 9 rendered as an HTML table.

## 5. Methods

- Brief, prose, no equations (those live in Theory).
- Lattice + parameter defaults (L = 64, Q = 48, kT = 0.5, default
  C_bulk swept).
- One MCS = L^2 site visits, then one diffusion step, then one
  segregation relaxation.
- Mention the propose-from-neighbors variant (efficiency note).
- Mention reproducibility: seeded RNG, full campaign in 35-60 s on a
  laptop.

## 6. Validation

- Source: `docs/testing_and_validation.md` Section 4.2 and
  `docs/final_report.md` Sections 4.1, 4.4, 4.8.
- Three subsections, each with one figure:
  - 6.1 Parabolic growth law -- figure
        `results/figures/exp1_d2_vs_t.png`. Caption: k = 1.10,
        R^2 = 0.96; same order as published Q-state Potts values.
  - 6.2 Late-time grain size distribution -- figure
        `results/figures/exp2_size_distribution.png`. Caption:
        unimodal, peak near D/<D> = 1, qualitatively Hillert-like;
        note the L = 64 statistics caveat.
  - 6.3 Solute drag curve -- figure
        `results/figures/exp4_drag_curve.png`. Caption: binned mean
        rises to a peak at intermediate v -- the Cahn signature.

## 7. Results: dopant sweep

- Two figures side by side (or stacked on mobile):
  - 7.1 Mean grain diameter vs MCS for the C_bulk sweep -- figure
        `results/figures/exp3_concentration_sweep.png`.
  - 7.2 Design curve: final <D> vs C_bulk -- figure
        `results/figures/exp6_design_curve.png`.
- Brief prose: more dopant -> slower growth -> smaller final grain
  size. C -> 0 limit recovers pure-growth <D> within finite-size
  noise (compare 0.000, 0.001, 0.010 = all ~14).
- Optional third figure: optical attenuation vs C_bulk
  `results/figures/exp7_attenuation_vs_concentration.png` -- bridges
  to the showcase section.

## 8. The showcase visualization

- The centerpiece. Largest figure on the page.
- Source: `results/figures/showcase_figure.png` (3 columns:
  C = 0.05 / 0.20 / 0.40; top row microstructure, bottom row
  scattering field, shared color scale, per-column <D> and mu_eff
  labels).
- Caption is the careful one already in
  `docs/final_report.md` Section 5 -- copy verbatim.
- Three short paragraphs after the figure walking through the low /
  medium / high dopant cases (cribbed from the same Section 5
  discussion).

## 9. Discussion

- Source: `docs/final_report.md` Section 7.
- Three subsections:
  - 9.1 The activator-vs-transparency trade-off -- two-sentence
        framing of the central finding.
  - 9.2 What grain size should a designer target? -- Rayleigh vs
        geometric crossover; numbers for Lu2O3:Eu at ~610 nm.
  - 9.3 Quantitative caveats -- what we can and cannot claim.
- No new figures.

## 10. Limitations

- Source: `docs/final_report.md` Section 9 (and
  `docs/testing_and_validation.md` Section 4.3).
- Bulleted list, six items: 2D, static drag coefficient, isotropic
  boundary energy, no stochastic solute noise, finite-size effects,
  illustrative-only optical proxy.
- Each item is one sentence.
- Closing line: "Quantitative optical predictions would require a
  coupled Monte Carlo photon-transport simulation, outside the scope
  of this class project."

## 11. Code and reproducibility

- One paragraph plus a code block.
- Link: <https://github.com/Grahamalt/grain-growth-sim>
- Instructions:
  ```
  git clone https://github.com/Grahamalt/grain-growth-sim.git
  cd grain-growth-sim
  pip install numpy scipy matplotlib pytest
  python code/main.py        # ~35-60 s, regenerates all figures
  python -m pytest tests/    # 67 tests
  ```
- Note: every figure on this page is regenerated by `code/main.py`
  from the seeded RNG and the parameters in `code/experiments.py`.

## 12. References

- Source: `docs/final_report.md` Section 10 (7 entries).
- Hillert 1965, Cahn 1962, Lucke and Stuwe 1971, Anderson et al.
  1984, Holm and Battaile 2001, Burke and Turnbull 1952, Ikesue
  and Aung 2008.
- Note in the section header: reference values quoted in Validation
  are cited from background knowledge, not re-checked for this
  writeup.

## 13. Footer

- Course: 3.21 (Kinetic Processes in Materials), Spring 2026.
- Date of last build / regeneration of figures (auto-stamp at build
  time if the page generator supports it).
- Contact: graham@altschuler.com.
- Repo link repeated as a small line.

---

## Build notes (not displayed on page)

- Planned format: single static HTML page (or single Markdown file
  rendered through a simple template). MathJax CDN for equations.
- Image strategy: link to PNGs already in `results/figures/`. No
  re-export needed -- they are already 300 DPI from Phase 8.0.
- Animation strategy (optional): assemble snapshots into a GIF with
  imageio or imagemagick:
  ```
  magick -delay 150 -loop 0 results/figures/snapshots/snapshot_*.png \
         results/figures/snapshots/grain_growth.gif
  ```
- Color choices: keep `tab20` for orientations and `inferno` for
  scattering on the page so they match the figures themselves.
- Update this outline as the page evolves; it is a working doc.
