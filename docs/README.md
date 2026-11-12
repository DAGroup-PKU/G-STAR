# G-STAR project-page draft

Project page for **G-STAR: Group-Scaled Token Capacity for Deeply Compressed Autoregressive Generation**.

## Preview locally

From the repository root:

```sh
python3 -m http.server 8765 --directory docs
```

Open http://localhost:8765/ in a browser. No build step or external runtime dependencies are needed.

## Contents

- `docs/index.html`: paper content and layout
- `docs/styles.css`: dark visual theme
- `docs/app.js`: result filters, image viewer, co-scaling demo, and editing comparisons
- `docs/assets/`: figures and samples extracted from the paper
- `docs/fonts/`: locally served fonts
- `docs/installation-and-usage.md`: installation, inference, data and training guide linked from the repository README

## Draft status

Figures and results use the local 18-page manuscript snapshot saved September 26, 2026. Title and abstract were checked against the supplied Overleaf project. Authors remain anonymous; public paper, code, weights, and citation metadata are pending. The paper button currently opens the private Overleaf project.

This repository remains private. Pushing these files does not enable GitHub Pages or change repository visibility. The `docs/` directory is ready for a later hosting configuration.
