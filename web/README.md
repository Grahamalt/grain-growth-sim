# Web page

Static project page for `grain-growth-sim`. The page is a single
`index.html` plus `style.css` plus `script.js`, with figures in
`assets/figures/`.

## Layout

```
web/
├── index.html
├── style.css
├── script.js
├── assets/
│   ├── figures/
│   │   ├── parabolic_growth.png
│   │   ├── grain_size_distribution.png
│   │   ├── concentration_sweep.png
│   │   ├── cahn_drag_curve.png
│   │   ├── segregation_energy_sweep.png
│   │   ├── design_curve.png
│   │   ├── attenuation_vs_concentration.png
│   │   ├── evolution_panels.png
│   │   ├── showcase_3x2.png
│   │   └── snapshots/
│   │       ├── snapshot_00_mcs0000.png
│   │       ├── snapshot_01_mcs0020.png
│   │       ├── snapshot_02_mcs0100.png
│   │       └── snapshot_03_mcs0200.png
│   └── animations/        # optional GIF/MP4 derived from snapshots/
└── README.md              # this file
```

All figures under `assets/figures/` are copies of the artefacts in the
top-level `results/figures/` directory, renamed for descriptive URLs.
Regenerate the originals with `python code/main.py` from the repo
root and re-copy if anything changes.

## Local preview

```sh
cd web
python3 -m http.server 8000
# open http://localhost:8000
```

## Optional: build the animation

To assemble the four snapshot frames into a looping GIF:

```sh
magick -delay 150 -loop 0 \
       assets/figures/snapshots/snapshot_*.png \
       assets/animations/grain_growth.gif
```

(or with `imageio`)

```python
import imageio.v3 as iio, glob
frames = [iio.imread(p) for p in sorted(glob.glob("assets/figures/snapshots/snapshot_*.png"))]
iio.imwrite("assets/animations/grain_growth.gif", frames, duration=1500, loop=0)
```

## Deployment

The page is fully static — any host that serves files works.

- **GitHub Pages:** in repo settings, set Pages to deploy from the
  `main` branch, folder `/web`. The page will be live at
  `https://Grahamalt.github.io/grain-growth-sim/`.
- **Netlify / Vercel / Cloudflare Pages:** point the project at this
  repo with publish directory `web/` and no build command.
- **Any static host:** upload the contents of `web/` as-is.
