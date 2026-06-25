import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

class RegularBin:
    """
    A class to define and manage multi-dimensional bins for data analysis.
    Bins can be defined either as a uniform grid or by providing explicit,
    non-uniform bin edges for each dimension.
    """

    def __init__(self, data=None, min_values=None, max_values=None, n_bins=None,
                 bin_edges=None, target=None, wall=False):
        """
        Args:
            data (np.ndarray, optional): Data to automatically determine bin ranges.
            min_values (list, optional): Minimum values for each dimension (for uniform grid).
            max_values (list, optional): Maximum values for each dimension (for uniform grid).
            n_bins (list, optional): Number of bins for each dimension (for uniform grid).
            bin_edges (list of lists/np.ndarrays, optional): A list where each element is a
                sorted list/array of bin edges for that dimension. e.g., [[0, 1, 3], [0, 2, 4]].
            target (list, optional): Coordinates of a target point for distance calculations.
            wall (bool): If True, points outside the grid are assigned to the nearest edge bin.
        """
        is_uniform_grid_defined = any(v is not None for v in [min_values, max_values, n_bins])
        if bin_edges is not None and is_uniform_grid_defined:
            raise ValueError("Provide EITHER 'bin_edges' OR uniform grid parameters ('min_values', 'max_values', 'n_bins'), not both.")
        if bin_edges is None and not is_uniform_grid_defined:
            raise ValueError("You must define the bins using either 'bin_edges' or uniform grid parameters.")

        self.data = data
        self.min_values = min_values
        self.max_values = max_values
        self.n_bins = n_bins
        self.bin_edges = [np.array(edges) for edges in bin_edges] if bin_edges is not None else None
        self.target = target
        self.wall = wall
        self.bin_widths = None

    def initialize_bins(self):
        """Prepares the bin parameters based on the initialization method."""
        if self.bin_edges is not None:
            self._initialize_from_edges()
        else:
            self._initialize_from_uniform_grid()

        self.all_bins = np.prod(self.n_bins).item()
        # MODIFICATION: The 'bin' column will now contain flat integer indices.
        self.df = pd.DataFrame(self.bin_centers.items(), columns=["bin_index", "bin_centers"])
        if self.target is not None:
            self.distance_from_target_bin()

        # Build sorted neighbor lists for each bin
        self._build_neighbor_lists()


    def _initialize_from_edges(self):
        """Initializes bin properties from user-provided bin edges."""
        print("Initializing bins from provided edges.")
        self.n_bins = [len(edges) - 1 for edges in self.bin_edges]
        self.min_values = [edges[0] for edges in self.bin_edges]
        self.max_values = [edges[-1] for edges in self.bin_edges]

        self.bin_centers = {}
        all_indices = [range(n) for n in self.n_bins]
        for indices in itertools.product(*all_indices):
            center_coords = [
                (self.bin_edges[i][idx] + self.bin_edges[i][idx + 1]) / 2
                for i, idx in enumerate(indices)
            ]
            # MODIFICATION: Key the dictionary by the flat bin index.
            flat_index = np.ravel_multi_index(indices, self.n_bins)
            self.bin_centers[flat_index] = np.array(center_coords)

    def _initialize_from_uniform_grid(self):
        """Initializes bin properties for a uniform grid."""
        print("Initializing bins as a uniform grid.")
        if self.data is not None and (self.min_values is None or self.max_values is None):
            data_min = np.min(self.data, axis=0)
            data_max = np.max(self.data, axis=0)
            buffer = 0.1 * (data_max - data_min)
            self.min_values = data_min - buffer if self.min_values is None else self.min_values
            self.max_values = data_max + buffer if self.max_values is None else self.max_values

        if self.min_values is None or self.max_values is None or self.n_bins is None:
            raise ValueError("For uniform grids, 'min_values', 'max_values', and 'n_bins' must be defined or inferable from data.")

        self.bin_widths = [(mx - mn) / n for mn, mx, n in zip(self.min_values, self.max_values, self.n_bins)]

        self.bin_centers = {}
        all_indices = [range(n) for n in self.n_bins]
        for indices in itertools.product(*all_indices):
            center_coords = [
                self.min_values[i] + idx * self.bin_widths[i] + self.bin_widths[i] / 2
                for i, idx in enumerate(indices)
            ]
            # MODIFICATION: Key the dictionary by the flat bin index.
            flat_index = np.ravel_multi_index(indices, self.n_bins)
            self.bin_centers[flat_index] = np.array(center_coords)

    def find_bin(self, data_point):
        """
        Determine the single, flat integer index for a given data point's bin.
        """
        data_point = np.asarray(data_point)
        if data_point.ndim != 1 or len(data_point) != len(self.n_bins):
            raise ValueError("data_point must be 1D array with length equal to number of dimensions")

        dim = data_point.shape[-1]

        # First, find the multi-dimensional grid coordinates
        if self.bin_edges is not None:
            grid_points = [
                np.searchsorted(self.bin_edges[i], data_point[i], side='right') - 1
                for i in range(dim)
            ]
        else:
            grid_points = [
                (data_point[i] - self.min_values[i]) / self.bin_widths[i]
                for i in range(dim)
            ]

        grid_points = np.array(grid_points, dtype=int)

        if self.wall:
            grid_points = self.adjust_bin_index(grid_points)

        flat_index = np.ravel_multi_index(grid_points, self.n_bins)
        return flat_index

    def plot_grid(self, ax1=0, ax2=1):
        """Visualize a 2D projection of the grid on a plot."""
        fig, ax = plt.subplots(figsize=(8, 8))

        if self.bin_edges is not None:
            for edge in self.bin_edges[ax1]:
                ax.axvline(x=edge, ls="--", c="red", lw=0.7)
            for edge in self.bin_edges[ax2]:
                ax.axhline(y=edge, ls="--", c="red", lw=0.7)
        else:
            for i in range(self.n_bins[ax1] + 1):
                ax.axvline(x=self.min_values[ax1] + i * self.bin_widths[ax1], ls="--", c="red", lw=0.7)
            for i in range(self.n_bins[ax2] + 1):
                ax.axhline(y=self.min_values[ax2] + i * self.bin_widths[ax2], ls="--", c="red", lw=0.7)

        ax.set_xlim(self.min_values[ax1], self.max_values[ax1])
        ax.set_ylim(self.min_values[ax2], self.max_values[ax2])
        ax.set_xlabel(f"Dimension {ax1}")
        ax.set_ylabel(f"Dimension {ax2}")
        ax.set_title("Binning Scheme")
        return fig, ax

    def adjust_bin_index(self, x):
        """Clamps the bin index to be within the valid range."""
        return np.clip(x, 0, np.array(self.n_bins) - 1)

    def distance_from_target_bin(self):
        """Calculates normalized inverse distance from each bin center to the target bin."""
        self.target = np.array(self.target)
        # MODIFICATION: find_bin now returns a flat index directly.
        self.target_bin_index = self.find_bin(self.target)
        target_bin_center = self.df[self.df.bin_index == self.target_bin_index].bin_centers

        self.df['distance_to_target'] = self.df['bin_centers'].apply(
            lambda x: np.linalg.norm(x - target_bin_center.values[0])
        )
        inv_r = self.df['distance_to_target'].apply(lambda x: 1/x if x != 0 else 0)
        # Normalize to a 0-1 range
        min_inv_r, max_inv_r = inv_r.min(), inv_r.max()
        if max_inv_r > min_inv_r:
            inv_r = (inv_r - min_inv_r) / (max_inv_r - min_inv_r)
        self.df['distance_to_target'] = inv_r

    def calculate_population(self, data, **kwargs):
        """Calculates the population and data indices for each bin."""
        # MODIFICATION: Initialize dictionaries using the flat bin index.
        population_dict = {item: 0 for item in self.df['bin_index'].values}
        population_data_dict = {item: [] for item in self.df['bin_index'].values}

        for index, points in enumerate(data):
            # MODIFICATION: find_bin now returns a flat index, no tuple conversion needed.
            flat_bin_index = self.find_bin(points)
            if flat_bin_index in population_dict:
                population_dict[flat_bin_index] += 1
                population_data_dict[flat_bin_index].append(index)

        self.df['population'] = self.df['bin_index'].apply(lambda x: population_dict[x])
        self.df['populated_data'] = self.df['bin_index'].apply(lambda x: population_data_dict[x])

    def scatter_plot(self, data, ax1, ax2, alpha=0.85):
        """Creates a scatter plot of data overlaid on the bin grid."""
        fig, ax = self.plot_grid(ax1, ax2)
        sns.scatterplot(x=data[:, ax1], y=data[:, ax2], s=10, alpha=alpha, ax=ax)
        return fig, ax

    def _build_neighbor_lists(self):
        """Precompute sorted neighbor bins for each bin."""
        self.neighbors = {}
        bin_indices = list(self.bin_centers.keys())
        centers = np.array(list(self.bin_centers.values()))

        for i, bin_idx in enumerate(bin_indices):
            center_i = centers[i]
            # Compute distances to all other bins
            distances = np.linalg.norm(centers - center_i, axis=1)
            # Sort indices by distance (ignore self)
            sorted_neighbors = [
                bin_indices[j] for j in np.argsort(distances) if j != i
            ]
            self.neighbors[bin_idx] = sorted_neighbors

    def get_sorted_neighbors(self, bin_index):
        """Return neighbor bins sorted by distance from given bin."""
        return self.neighbors.get(bin_index, [])

# --- Main execution block to demonstrate the functionality ---
if __name__ == "__main__":
    np.random.seed(42)
    sample_data = np.random.randn(500, 2)

    print("--- DEMO 1: Uniform Grid ---")
    uniform_bins = RegularBin(min_values=[-3, -3], max_values=[3, 3], n_bins=[5, 10], wall=True)
    uniform_bins.initialize_bins()
    fig1, ax1 = uniform_bins.plot_grid(ax1=0, ax2=1)
    ax1.set_title("Uniform Grid (5x10)")
    ax1.scatter(sample_data[:, 0], sample_data[:, 1], alpha=0.5, s=10)
    plt.show()

    point1 = np.array([0.1, -2.9])
    bin_idx1 = uniform_bins.find_bin(point1)
    print(f"Point {point1} is in uniform bin with flat index: {bin_idx1}\n") # Example: index 20

    print("\n" + "="*50 + "\n")

    print("--- DEMO 2: Non-Uniform Grid via bin_edges ---")
    x_edges = [-3, -1, -0.5, 0.5, 1, 3]  # 5 bins
    y_edges = [-3, -2, 0, 2, 3]         # 4 bins
    non_uniform_bins = RegularBin(bin_edges=[x_edges, y_edges], wall=True)
    non_uniform_bins.initialize_bins()

    print(f"Number of bins derived from edges: {non_uniform_bins.n_bins}")

    fig2, ax2 = non_uniform_bins.plot_grid(ax1=0, ax2=1)
    ax2.set_title("Non-Uniform Grid via Bin Edges")
    ax2.scatter(sample_data[:, 0], sample_data[:, 1], alpha=0.5, s=10)
    plt.show()

    point2 = np.array([0.1, -2.9])
    bin_idx2 = non_uniform_bins.find_bin(point2)
    print(f"Point {point2} is in non-uniform bin with flat index: {bin_idx2}") # Example: index 8
