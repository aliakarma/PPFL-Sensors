import unittest
from types import SimpleNamespace

import torch
from unittest.mock import MagicMock

from attack.inference_attack import GradientInferenceAttack
from client.client import ClientUpdate

class TestNoLeakage(unittest.TestCase):
    def setUp(self):
        # Create dummy config and tracker
        self.config = SimpleNamespace(
            attack=SimpleNamespace(
                enabled=True,
                collect_rounds=5,
                eval_start_round=7,
                model=["mlp"],
                grad_norm="l2",
                pca_components=0,
                mlp_epochs=1,
            ),
            dataset=SimpleNamespace(n_clients=2),
            logging=SimpleNamespace(seed=42)
        )

        self.tracker = MagicMock()
        self.tracker.gradient_store = MagicMock()

        # Dummy dataset
        self.tracker.gradient_store.get_train_dataset.return_value = (
            torch.randn(10, 100), torch.randint(0, 2, (10,))
        )

        self.attack = GradientInferenceAttack(self.config, self.tracker)

    def test_strict_phase_separation_and_leakage(self):
        collect_rounds = self.attack._collect_rounds
        train_round = collect_rounds + 1
        eval_start_round = self.attack._eval_start

        self.assertTrue(train_round > collect_rounds)
        self.assertTrue(eval_start_round > train_round)

        # Simulate rounds 1 through 10
        for round_idx in range(1, 11):
            # Create mock updates
            updates = [
                ClientUpdate(client_id=0, defended_gradients=torch.randn(100), num_samples=100),
                ClientUpdate(client_id=1, defended_gradients=torch.randn(100), num_samples=100),
            ]

            if round_idx <= collect_rounds:
                self.attack.collect(round_idx, updates)
            elif round_idx == train_round:
                self.attack.train()
            elif round_idx >= eval_start_round:
                self.attack.evaluate(round_idx, updates)

        # Ensure train_ids is disjoint from eval_ids
        self.assertTrue(set(self.attack._train_ids).isdisjoint(set(self.attack._eval_ids)))
        self.assertEqual(len(self.attack._train_ids), 10)  # 5 rounds * 2 clients
        self.assertEqual(len(self.attack._eval_ids), 8)    # 4 rounds (7-10) * 2 clients

    def test_prevent_eval_store(self):
        # Attempt to store gradient from an evaluation round
        update = [ClientUpdate(client_id=0, defended_gradients=torch.randn(100), num_samples=100)]
        self.attack.collect(10, update) # Since collect_rounds=5
        
        self.assertEqual(len(self.attack._train_ids), 0)

if __name__ == '__main__':
    unittest.main()
