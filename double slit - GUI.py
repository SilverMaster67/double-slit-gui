import tkinter as tk
from tkinter import ttk, filedialog
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ---------------------------------------------------
# Analytic double-slit + single-slit patterns
# ---------------------------------------------------

def compute_patterns(lambda_nm, d_um, w_um, L_m,
                     waist_mm=5.0,
                     xmax=0.02, nbins=800):
    """
    Compute realistic double-slit patterns (interference, which-way, single-slit)
    for given parameters.

    Parameters
    ----------
    lambda_nm : float
        Wavelength in nanometers.
    d_um : float
        Slit separation in micrometers.
    w_um : float
        Slit width in micrometers.
    L_m : float
        Screen distance in meters.
    waist_mm : float
        1/e beam half-width in millimeters (Gaussian beam envelope).
    xmax : float
        Half-width of screen window in meters.
    nbins : int
        Number of x-points.

    Returns
    -------
    x_mm : ndarray
        Screen positions in millimeters.
    I_interf : ndarray
        Normalized interference intensity.
    I_which : ndarray
        Normalized which-way intensity.
    I_single : ndarray
        Normalized single-slit intensity (one open).
    fringe_mm : float
        Fringe spacing in millimeters (for double-slit).
    """
    # Units
    lambda0 = lambda_nm * 1e-9   # [m]
    d = d_um * 1e-6              # [m]
    w = w_um * 1e-6              # [m]
    L = L_m                      # [m]
    waist = waist_mm * 1e-3      # [m]

    x = np.linspace(-xmax, xmax, nbins)  # [m]
    theta = np.arctan2(x, L)
    sin_theta = np.sin(theta)

    # Single-slit diffraction variable
    beta = np.pi * w * sin_theta / lambda0
    # Phase difference between slits
    delta_phi = 2 * np.pi * d * sin_theta / lambda0

    # sinc envelope: sin(beta)/beta = sinc(beta/pi)
    single_slit_envelope = np.sinc(beta / np.pi)

    # Gaussian beam envelope
    if waist > 0:
        gauss_env = np.exp(-(x / waist) ** 2)
    else:
        gauss_env = 1.0

    # Total single-slit amplitude
    amp_single = single_slit_envelope * gauss_env
    I_single = amp_single ** 2

    # Coherent double-slit
    I_interf = 4 * I_single * np.cos(delta_phi / 2) ** 2

    # Which-way (incoherent sum of two single-slits)
    I_which = 2 * I_single

    # Normalize each (avoid division by zero)
    if I_interf.max() > 0:
        I_interf = I_interf / I_interf.max()
    if I_which.max() > 0:
        I_which = I_which / I_which.max()
    if I_single.max() > 0:
        I_single = I_single / I_single.max()

    # Fringe spacing λL/d (in meters -> mm)
    if d > 0:
        fringe = lambda0 * L / d * 1e3  # [mm]
    else:
        fringe = np.nan

    return x * 1e3, I_interf, I_which, I_single, fringe


# ---------------------------------------------------
# Simple Monte Carlo MUM-like simulation
# ---------------------------------------------------

def mc_step(lambda_nm, d_um, w_um, L_m,
            xmax=0.02, nbins=400,
            N_step=2000,
            sigma_theta=0.002,
            rng=None,
            A_total=None):
    """
    One Monte-Carlo step: sample N_step monopole trajectories and
    accumulate complex amplitudes in screen bins (coherent double-slit).

    Returns updated arrays (x_mm, I_mc, A_total).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Units & geometry (for slits)
    lambda0 = lambda_nm * 1e-9
    k = 2 * np.pi / lambda0
    d = d_um * 1e-6
    w = w_um * 1e-6
    L = L_m
    Y0 = L  # just take same order for source distance

    # Slits centered at +/- d/2
    x1_center = -d / 2
    x2_center = d / 2
    x1_min, x1_max = x1_center - w / 2, x1_center + w / 2
    x2_min, x2_max = x2_center - w / 2, x2_center + w / 2

    # Screen grid
    x = np.linspace(-xmax, xmax, nbins)
    bin_edges = np.linspace(-xmax, xmax, nbins + 1)

    if A_total is None:
        A_total = np.zeros(nbins, dtype=np.complex128)

    def find_bin_index(x_val):
        if x_val < bin_edges[0] or x_val >= bin_edges[-1]:
            return -1
        kbin = np.searchsorted(bin_edges, x_val) - 1
        if kbin < 0 or kbin >= nbins:
            return -1
        return kbin

    for _ in range(N_step):
        # initial direction from small angular spread
        theta = rng.normal(loc=0.0, scale=sigma_theta)
        dir_x = np.sin(theta)
        dir_y = np.cos(theta)
        if dir_y <= 0:
            continue

        # source at (0, -Y0) -> slits plane y=0
        t_s = Y0 / dir_y
        x_s = 0.0 + t_s * dir_x

        in_slit1 = (x1_min <= x_s <= x1_max)
        in_slit2 = (x2_min <= x_s <= x2_max)
        if not (in_slit1 or in_slit2):
            continue  # hit barrier

        # propagate to screen y=L
        tau = L / dir_y
        x_f = x_s + tau * dir_x

        # total path length ≈ two straight segments
        ds1 = np.hypot(x_s, Y0)
        ds2 = np.hypot(x_f - x_s, L)
        ell = ds1 + ds2
        phase = k * ell
        w_c = np.exp(1j * phase)

        kbin = find_bin_index(x_f)
        if kbin == -1:
            continue

        A_total[kbin] += w_c

    I_mc = np.abs(A_total) ** 2
    if I_mc.max() > 0:
        I_mc = I_mc / I_mc.max()

    return x * 1e3, I_mc, A_total


# ---------------------------------------------------
# Tkinter GUI
# ---------------------------------------------------

class DoubleSlitGUI:
    def __init__(self, master):
        self.master = master
        master.title("MUM Double-Slit Interference")

        # ---------- State ----------
        self.lambda_nm = tk.DoubleVar(value=810.0)
        self.d_um      = tk.DoubleVar(value=100.0)
        self.w_um      = tk.DoubleVar(value=20.0)
        self.L_m       = tk.DoubleVar(value=1.0)
        self.waist_mm  = tk.DoubleVar(value=5.0)

        self.mode_var  = tk.StringVar(value="analytic")   # 'analytic' or 'mc'
        self.show_heat = tk.BooleanVar(value=True)
        self.show_single = tk.BooleanVar(value=True)

        self.mc_running = False
        self.mc_rng = np.random.default_rng()
        self.mc_A_total = None
        self.mc_x_mm = None
        self.mc_I = None

        # ---------- Layout ----------
        mainframe = ttk.Frame(master)
        mainframe.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(mainframe)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(mainframe)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # ---------- Matplotlib figure ----------
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            1, 2, figsize=(10, 4), dpi=100,
            gridspec_kw={'width_ratios': [2, 1]}
        )
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initial analytic plot
        x_mm, I_int, I_wh, I_single, fringe = compute_patterns(
            self.lambda_nm.get(),
            self.d_um.get(),
            self.w_um.get(),
            self.L_m.get(),
            waist_mm=self.waist_mm.get()
        )

        (self.line_interf,) = self.ax1.plot(
            x_mm, I_int, label="coherent double-slit"
        )
        (self.line_which,) = self.ax1.plot(
            x_mm, I_wh, "--", label="which-way"
        )
        (self.line_single,) = self.ax1.plot(
            x_mm, I_single, ":", label="single slit"
        )

        self.ax1.set_xlabel("x on screen [mm]")
        self.ax1.set_ylabel("Normalized intensity")
        self.ax1.set_title("MUM: symplectic monopoles + contact events")
        self.ax1.grid(alpha=0.2)
        self.ax1.legend(loc="upper right")
        self.ax1.set_ylim(0, 1.05)

        # Heatmap on ax2
        self.heatmap = None
        self.update_heatmap(x_mm, I_int)

        self.canvas.draw()

        # Fringe label
        self.fringe_label = ttk.Label(right_frame, text="")
        self.fringe_label.pack(pady=(5, 5))

        # ---------- Sliders / controls ----------
        ttk.Label(right_frame, text="Wavelength λ [nm]").pack(pady=(5, 0))
        tk.Scale(
            right_frame, from_=400, to=1600, orient=tk.HORIZONTAL,
            resolution=10, variable=self.lambda_nm,
            command=lambda v: self.update_all()
        ).pack(fill=tk.X, padx=10)

        ttk.Label(right_frame, text="Slit separation d [μm]").pack(pady=(5, 0))
        tk.Scale(
            right_frame, from_=20, to=300, orient=tk.HORIZONTAL,
            resolution=5, variable=self.d_um,
            command=lambda v: self.update_all()
        ).pack(fill=tk.X, padx=10)

        ttk.Label(right_frame, text="Slit width w [μm]").pack(pady=(5, 0))
        tk.Scale(
            right_frame, from_=5, to=100, orient=tk.HORIZONTAL,
            resolution=5, variable=self.w_um,
            command=lambda v: self.update_all()
        ).pack(fill=tk.X, padx=10)

        ttk.Label(right_frame, text="Screen distance L [m]").pack(pady=(5, 0))
        tk.Scale(
            right_frame, from_=0.2, to=5.0, orient=tk.HORIZONTAL,
            resolution=0.1, variable=self.L_m,
            command=lambda v: self.update_all()
        ).pack(fill=tk.X, padx=10)

        ttk.Label(right_frame, text="Beam waist [mm]").pack(pady=(5, 0))
        tk.Scale(
            right_frame, from_=0.5, to=20.0, orient=tk.HORIZONTAL,
            resolution=0.5, variable=self.waist_mm,
            command=lambda v: self.update_all()
        ).pack(fill=tk.X, padx=10, pady=(0, 10))

        # Mode selection
        ttk.Label(right_frame, text="Mode").pack(pady=(5, 0))
        mode_frame = ttk.Frame(right_frame)
        mode_frame.pack(fill=tk.X, padx=10)
        tk.Radiobutton(
            mode_frame, text="Analytic", value="analytic",
            variable=self.mode_var, command=self.on_mode_change
        ).pack(side=tk.LEFT)
        tk.Radiobutton(
            mode_frame, text="Monte Carlo", value="mc",
            variable=self.mode_var, command=self.on_mode_change
        ).pack(side=tk.LEFT)

        # Checkbuttons
        tk.Checkbutton(
            right_frame, text="Show single-slit curve",
            variable=self.show_single, command=self.update_visibility
        ).pack(pady=(5, 0))
        tk.Checkbutton(
            right_frame, text="Show heatmap",
            variable=self.show_heat, command=self.update_heat_visibility
        ).pack(pady=(0, 10))

        # MC controls
        self.mc_button = ttk.Button(
            right_frame, text="Start MC animation",
            command=self.toggle_mc
        )
        self.mc_button.pack(pady=(5, 0))

        ttk.Button(
            right_frame, text="Reset MC", command=self.reset_mc
        ).pack(pady=(5, 10))

        # Save button
        ttk.Button(
            right_frame, text="Save PNG", command=self.save_png
        ).pack(pady=(5, 10))

        # Manual update
        ttk.Button(
            right_frame, text="Update", command=self.update_all
        ).pack(pady=(5, 10))

        # Initial fringe label
        self.update_fringe_label(fringe)

    # ---------- Plot helpers ----------

    def update_fringe_label(self, fringe_mm):
        if np.isnan(fringe_mm):
            txt = "Fringe spacing: n/a"
        else:
            txt = f"Fringe spacing Δx ≈ {fringe_mm:.2f} mm"
        self.fringe_label.config(text=txt)

    def update_heatmap(self, x_mm, I_interf):
        self.ax2.clear()
        if self.show_heat.get():
            # simple 2D stripe pattern: replicate 1D interference along y
            ny = 200
            y = np.linspace(-20, 20, ny)
            I2D = np.outer(np.ones_like(y), I_interf)
            extent = [x_mm[0], x_mm[-1], y[0], y[-1]]
            self.ax2.imshow(
                I2D.T, origin="lower", aspect="auto", extent=extent,
                cmap="viridis"
            )
            self.ax2.set_xlabel("x [mm]")
            self.ax2.set_ylabel("y (arbitrary)")
            self.ax2.set_title("2D interference (schematic)")
        else:
            self.ax2.set_axis_off()

    def update_heat_visibility(self):
        # just recompute analytic and heatmap
        self.update_all()

    def update_visibility(self):
        # single-slit checkbox toggles line visibility
        self.line_single.set_visible(self.show_single.get())
        self.canvas.draw_idle()

    # ---------- Mode handling ----------

    def on_mode_change(self):
        # Stop MC animation if leaving MC mode
        if self.mode_var.get() != "mc":
            self.mc_running = False
            self.mc_button.config(text="Start MC animation")
        self.update_all()

    # ---------- Main update ----------

    def update_all(self):
        mode = self.mode_var.get()

        if mode == "analytic":
            # analytic curves
            x_mm, I_int, I_wh, I_single, fringe = compute_patterns(
                self.lambda_nm.get(),
                self.d_um.get(),
                self.w_um.get(),
                self.L_m.get(),
                waist_mm=self.waist_mm.get()
            )
            self.line_interf.set_data(x_mm, I_int)
            self.line_which.set_data(x_mm, I_wh)
            self.line_single.set_data(x_mm, I_single)

            self.line_interf.set_label("coherent double-slit")
            self.line_which.set_label("which-way")
            self.ax1.legend(loc="upper right")

            self.ax1.set_xlim(x_mm[0], x_mm[-1])
            self.ax1.set_ylim(0, 1.05)

            self.update_heatmap(x_mm, I_int)
            self.update_fringe_label(fringe)

        else:
            # Monte Carlo mode: show MC distribution vs analytic envelope
            if self.mc_x_mm is None or self.mc_I is None:
                # run a small initial MC to have something
                self.reset_mc()
                self.mc_step_once()

            # analytic envelope (for comparison) – just single call
            x_mm, I_int, _, _, fringe = compute_patterns(
                self.lambda_nm.get(),
                self.d_um.get(),
                self.w_um.get(),
                self.L_m.get(),
                waist_mm=self.waist_mm.get(),
                xmax=self.mc_x_mm.max() / 1e3,
                nbins=len(self.mc_x_mm)
            )

            self.line_interf.set_data(self.mc_x_mm, self.mc_I)
            self.line_interf.set_label("MC MUM (coherent)")

            self.line_which.set_data(x_mm, I_int)
            self.line_which.set_label("analytic envelope")

            # single-slit off in MC mode (optional)
            self.line_single.set_data([], [])
            self.ax1.legend(loc="upper right")

            self.ax1.set_xlim(self.mc_x_mm[0], self.mc_x_mm[-1])
            self.ax1.set_ylim(0, 1.05)

            self.update_heatmap(self.mc_x_mm, self.mc_I)
            self.update_fringe_label(fringe)

        self.update_visibility()
        self.canvas.draw_idle()

    # ---------- Monte Carlo controls ----------

    def reset_mc(self):
        # reset MC state
        self.mc_A_total = None
        self.mc_x_mm = None
        self.mc_I = None

    def mc_step_once(self):
        x_mm, I_mc, self.mc_A_total = mc_step(
            self.lambda_nm.get(),
            self.d_um.get(),
            self.w_um.get(),
            self.L_m.get(),
            xmax=0.02,
            nbins=400,
            N_step=2000,
            rng=self.mc_rng,
            A_total=self.mc_A_total
        )
        self.mc_x_mm = x_mm
        self.mc_I = I_mc

    def toggle_mc(self):
        if self.mode_var.get() != "mc":
            return
        self.mc_running = not self.mc_running
        if self.mc_running:
            self.mc_button.config(text="Stop MC animation")
            self.run_mc_animation()
        else:
            self.mc_button.config(text="Start MC animation")

    def run_mc_animation(self):
        if not self.mc_running:
            return
        # Do one MC step and update plot
        self.mc_step_once()
        self.update_all()
        # Schedule next frame
        self.master.after(50, self.run_mc_animation)

    # ---------- Save PNG ----------

    def save_png(self):
        # Default filename with timestamp
        default_name = f"mum_double_slit_{int(time.time())}.png"
        fname = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if fname:
            self.fig.savefig(fname, dpi=200)
            print(f"Saved figure to {fname}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DoubleSlitGUI(root)
    root.mainloop()