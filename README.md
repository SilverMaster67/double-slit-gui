# MUM Double-Slit Simulator

Interactive double-slit simulator inspired by the **Most Useless Machine (MUM)** framework.

The code provides a bridge between standard wave-optics and the MUM picture:

- **Symplectic monopoles** = classical trajectories + phases  
- **Contact events** = detection (dipole absorption)  
- Interference vs which-way is controlled by when/where contact happens.

The GUI lets you tweak physical parameters in real time and immediately see how the pattern changes.

## Features

- Analytic double-slit pattern with realistic parameters:
  - wavelength `λ` (nm)
  - slit separation `d` (µm)
  - slit width `w` (µm)
  - screen distance `L` (m)
  - Gaussian beam waist
- Three curves:
  - **coherent double-slit** (interference)
  - **which-way** (no interference)
  - **single slit** (envelope)
- Fringe spacing display:  
  \[
  \Delta x \approx \frac{\lambda L}{d}
  \]
- Simple 2D heatmap of the interference pattern.
- Monte-Carlo **MUM mode**:
  - samples symplectic trajectories (monopoles),
  - attaches a phase from geometric path length,
  - builds the pattern by summing complex amplitudes.
- Start/stop MC animation and reset.
- Save current figure as PNG.

## Installation

```bash
git clone https://github.com/SilverMaster67/double-slit-gui.git
cd double-slit-gui
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
