import unittest
import numpy as np

from data.dataset_loader import partition_dirichlet, partition_pathological, SyntheticLoader

class TestDataPartitioning(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        # Using synthetic data directly for unit testing
        self.X, self.y, _, _ = SyntheticLoader(n_samples=1000, n_features=100, n_classes=4, seed=42).load()
        self.n_clients = 3

    def _verify_disjointness_and_coverage(self, partitions, total_samples):
        all_samples = []
        for X_c, y_c in partitions:
            for x_val in X_c:
                all_samples.append(tuple(x_val.tolist()))
        
        # Coverage
        self.assertEqual(len(all_samples), total_samples)
        
        # Disjointness / No duplication
        unique_samples = set(all_samples)
        self.assertEqual(len(all_samples), len(unique_samples))

    def test_dirichlet_partition(self):
        alpha = 0.5
        partitions = partition_dirichlet(self.X, self.y, self.n_clients, alpha, self.rng)
        self._verify_disjointness_and_coverage(partitions, len(self.X))
        
        # Verify heterogeneity
        dist = []
        for _, y_c in partitions:
            unique, counts = np.unique(y_c, return_counts=True)
            dist.append(dict(zip(unique, counts)))
        
        # The distributions should not be perfectly equal if properly skewed
        self.assertNotEqual(dist[0], dist[1])

    def test_pathological_partition(self):
        classes_per_client = 2
        partitions = partition_pathological(self.X, self.y, self.n_clients, classes_per_client, self.rng)
        self._verify_disjointness_and_coverage(partitions, len(self.X))
        
        for _, y_c in partitions:
            unique = np.unique(y_c)
            # A client might get fewer if the pool exhausted but strictly <= classes_per_client
            self.assertLessEqual(len(unique), classes_per_client)

if __name__ == '__main__':
    unittest.main()