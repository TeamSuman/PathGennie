import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------
# Example: load your sampled x,y coordinates from simulation
# Replace these with actual data
# --------------------------------------------
# Example fake data:
# x = np.random.randn(100000)
# y = np.random.randn(100000)
# If you already have x and y from your simulation, just do:
# x, y = mytrajectory[:,0], mytrajectory[:,1]
# --------------------------------------------
# Use PyEMMA's plotting code (you already provided)
# --------------------------------------------
# We import only needed functions from your big module:
from .plot2d import _to_free_energy, get_histogram, plot_map
from scipy.interpolate import griddata


def plot_free_energy_with_annotation(x, y,
                                     annotate_points=None,
                                     nbins=150,
                                     kT=1.0, ax = None):
    """
    Compute free energy map from (x,y) samples and annotate given points.

    annotate_points: list of tuples (x,y,"label")
    """

    # ---------------------------------------------------------
    # Compute the free energy landscape
    # ---------------------------------------------------------
    Xgrid, Ygrid, hist = get_histogram(x, y, nbins=nbins, avoid_zero_count=True)

    F = _to_free_energy(hist, minener_zero=True) * kT

    # ---------------------------------------------------------
    # PLOT
    # ---------------------------------------------------------
    fig, ax, misc = plot_map(
        Xgrid, Ygrid, F,
        cmap="nipy_spectral",
        cbar=True, ax = ax,
        cbar_label="Free energy / kT"
    )

    # ---------------------------------------------------------
    # OPTIONAL: label contour lines
    # ---------------------------------------------------------
    # CS = misc['mappable']   # this is the contourf
    # Make black contour lines and label them
    contour_lines = ax.contour(Xgrid, Ygrid, F, 10, colors='black', linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.1f")

    # ---------------------------------------------------------
    # Annotate specific (x,y) points
    # ---------------------------------------------------------
    if annotate_points is not None:
        # Build interpolator on the grid
        XY_samples = np.vstack([Xgrid.flatten(), Ygrid.flatten()]).T
        F_samples = F.flatten()

        def interp_FE(xp, yp):
            """
            Return interpolated Free energy at point (xp, yp)
            """
            val = griddata(XY_samples, F_samples, (xp, yp), method='cubic')
            return float(val)

        for (xp, yp, label) in annotate_points:
            FE_val = interp_FE(xp, yp)
            ax.scatter(xp, yp, s=40, c='white', edgecolors='k', zorder=10)
            ax.text(xp, yp, f"{label}\nF={FE_val:.2f}",
                    color='k', fontsize=9,
                    ha='left', va='bottom', weight='bold')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    #ax.set_title("Free Energy Surface with annotations")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------

if __name__ == "__main__":

    # Dummy example data for testing
    # Replace with your simulation data!
    # x = np.random.normal(0, 1, 50000)
    # y = np.random.normal(0, 1, 50000)

    annotate = [
        (-1.0, 0.0, "Basin A"),
        (1.0, 0.0, "Basin B"),
        (0.0, 1.2, "Barrier"),
    ]

    #plot_free_energy_with_annotation(x, y, annotate_points=annotate)
