import numpy as np

class Resampler:
    def __init__(self, bins, target_per_bin=10, weight_threshold=1e-30, seed=None):
        self.bins = bins
        self.N_TARGET_PER_BIN = target_per_bin
        self.weight_threshold = weight_threshold
        # Resampling draws go through a dedicated Generator rather than NumPy's
        # global RNG, so a run can be reproduced from a single seed. seed=None
        # keeps the previous non-deterministic behaviour.
        self.rng = np.random.default_rng(seed)

    def assign_bins_and_weights(self, walkers):
        if not walkers:
            return np.array([]), np.array([])
        #for w in walkers:
        #    print(w.progress_coord)
        assignments = np.array([self.bins.find_bin(w.progress_coord) for w in walkers], dtype=int)
        weights = np.array([w.weight for w in walkers], dtype=float)
        return assignments, weights

    def resample(self, walkers, warp_idx = None):
        if not walkers:
            return [], {}

        # Filter by threshold
        #if warp_idx is not None:
            #final_walkers = [w for j, w in enumerate(walkers) if j not in warp_idxs]
            #final_walkers = [w for w in walkers if w.weight >= self.weight_threshold]
        final_walkers = walkers
        if not final_walkers:
            return [], {}

        assignments, weights = self.assign_bins_and_weights(final_walkers)
        occupied_bins = np.unique(assignments).astype(int)

        resampled_walkers = []
        resampled_to_original = {}  # mapping: new_index -> original_index/indices

        new_idx = 0
        for bin_id in occupied_bins:
            parent_indices = np.where(assignments == bin_id)[0]
            if warp_idx is not None:
                parent_indices = [idx for idx in parent_indices if idx not in warp_idx]
            if len(parent_indices) == 0:
                continue  # skip this bin entirely
            bin_weights = weights[parent_indices]

            if len(parent_indices) == self.N_TARGET_PER_BIN:
                # keep as-is
                for idx in parent_indices:
                    resampled_walkers.append(final_walkers[idx])
                    resampled_to_original[new_idx] = int(idx)  # one-to-one mapping
                    new_idx += 1

            elif len(parent_indices) < self.N_TARGET_PER_BIN:
                walkers_out, indices = self.split_walkers(parent_indices, bin_weights, final_walkers)
                for w, parent_idx in zip(walkers_out, indices):
                    resampled_walkers.append(w)
                    resampled_to_original[new_idx] = int(parent_idx)  # one-to-one from split parent
                    new_idx += 1

            else:  # too many
                walkers_out, merge_events = self.merge_walkers(parent_indices, bin_weights, final_walkers)
                for w in walkers_out:
                    resampled_walkers.append(w)
                # build mapping: survivors inherit from multiple parents
                for survivor, event in zip(walkers_out, merge_events):
                    resampled_to_original[new_idx] = event  # event already stores [survivor_idx, loser_idx]
                    new_idx += 1
        return resampled_walkers, resampled_to_original

    def split_walkers(self, walker_indices, weights, final_walkers):
        total_weight = np.sum(weights)
        number_of_clones = self.N_TARGET_PER_BIN - len(walker_indices)
        if total_weight > 0:
            prob = weights / total_weight
        else:
            prob = np.ones_like(weights) / len(weights)
        clone_indices = self.rng.choice(walker_indices, size=number_of_clones, replace=True, p=prob)
        split_events = []
        unique_parents, clone_counts = np.unique(clone_indices, return_counts=True)
        for parent_idx, num_clones in zip(unique_parents, clone_counts):
            split_events.append({'parent_idx': int(parent_idx), 'num_clones': int(num_clones)})
        updated_indices = np.concatenate([walker_indices, clone_indices])
        unique_indices, counts = np.unique(updated_indices, return_counts=True)
        idx_to_count = dict(zip(unique_indices, counts))
        resampled = []
        indices = []
        for idx in unique_indices:
            parent_walker = final_walkers[idx]
            num_children = idx_to_count[idx]
            child_weight = parent_walker.weight / num_children
            for _ in range(num_children):
                new_walker = parent_walker.clone()
                new_walker.weight = child_weight
                resampled.append(new_walker)
                indices.append(idx)
        return resampled, indices

    def merge_walkers(self, walker_indices, weights, final_walkers):
        """
        Pairwise merge walkers in a bin until N_TARGET_PER_BIN remain.
        Survivor chosen proportional to relative weights.
        Operates directly on existing walker objects (no cloning).
        """
        # Extract walkers from this bin
        bin_walkers = [final_walkers[i] for i in walker_indices]
        merge_event = []
        while len(bin_walkers) > self.N_TARGET_PER_BIN:
            # Randomly choose two distinct walkers
            i, j = self.rng.choice(len(bin_walkers), size=2, replace=False)
            w1, w2 = bin_walkers[i], bin_walkers[j]

            # Pick survivor proportional to weights
            total = w1.weight + w2.weight
            if total == 0:
                survivor, loser = (w1, w2) if self.rng.random() < 0.5 else (w2, w1)
            else:
                p_survive_w1 = w1.weight / total
                if self.rng.random() < p_survive_w1:
                    survivor, loser = (w1, w2)
                    merge_event.append([walker_indices[i], walker_indices[j]])
                else:
                    survivor, loser = (w2, w1)
                    merge_event.append([walker_indices[j], walker_indices[i]])

            # Transfer weight
            survivor.weight += loser.weight

            # Remove loser from the list
            bin_walkers.remove(loser)

        return bin_walkers, merge_event
    def cap_and_redistribute(self, walkers, max_weight):
        """
        Cap walker weights at max_weight and redistribute excess
        equally among other walkers in the same bin.
        """
        weights = np.array([w.weight for w in walkers], dtype=float)
        excess = 0.0

        # Cap weights and collect excess
        for i, w in enumerate(walkers):
            if weights[i] > max_weight:
                excess += weights[i] - max_weight
                weights[i] = max_weight

        # Redistribute excess equally to all walkers
        if len(walkers) > 0:
            share = excess / len(walkers)
            weights += share

        # Update walkers with new weights
        for i, w in enumerate(walkers):
            w.weight = weights[i]

        return walkers
    def normalize(self, walkers, max_weight):

        final_walkers = walkers
        if not final_walkers:
            return [], {}

        assignments, _ = self.assign_bins_and_weights(final_walkers)
        occupied_bins = np.unique(assignments).astype(int)
        all_walkers = []
        for bin_id in occupied_bins:
            parent_indices = np.where(assignments == bin_id)[0]

            if len(parent_indices) == 0:
                continue  # skip this bin entirely
            bin_walkers = [walkers[indx].clone() for indx in parent_indices]
            #bin_weights = weights[parent_indices]
            new_walkers = self.cap_and_redistribute(bin_walkers, max_weight)
            all_walkers.extend(new_walkers)
        return all_walkers
