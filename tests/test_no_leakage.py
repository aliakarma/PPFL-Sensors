import unittest
from collections import OrderedDict
from types import SimpleNamespace

import torch
from unittest.mock import MagicMock

from attack.inference_attack import GradientInferenceAttack
from client.client import ClientUpdate
from utils.experiment_tracker import GradientStore
import os

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
        # Initialize real GradientStore for genuine leakage test
        os.makedirs("test_artifacts", exist_ok=True)
        self.tracker.gradient_store = GradientStore(artifact_dir="test_artifacts", collect_rounds=5, storage_type="raw")

        # Dummy dataset
        self.tracker.gradient_store.get_train_dataset = MagicMock(return_value=(
            torch.randn(10, 100), torch.randint(0, 2, (10,))
        ))

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
                ClientUpdate(
                    client_id=0, 
                    defended_gradients=torch.randn(100), 
                    n_samples=100,
                    weight_delta=OrderedDict(),
                    raw_gradients=torch.randn(100),
                    local_loss=0.5
                ),
                ClientUpdate(
                    client_id=1, 
                    defended_gradients=torch.randn(100), 
                    n_samples=100,
                    weight_delta=OrderedDict(),
                    raw_gradients=torch.randn(100),
                    local_loss=0.5
                ),
            ]

            if round_idx <= collect_rounds:
                self.attack.collect(round_idx, updates)
            elif round_idx == train_round:
                self.attack.train()
            elif round_idx >= eval_start_round:
                self.attack.evaluate(round_idx, updates)

        # Hash-level disjointness inherently asserted via GradientStore._train_hashes and _eval_hashes
        self.assertTrue(len(self.tracker.gradient_store._train_hashes) > 0)
        self.assertTrue(len(self.tracker.gradient_store._eval_hashes) > 0)
        self.assertTrue(self.tracker.gradient_store._train_hashes.isdisjoint(self.tracker.gradient_store._eval_hashes))
        self.assertEqual(len(self.tracker.gradient_store._train_hashes), 10)  # 5 rounds * 2 clients
        self.assertEqual(len(self.tracker.gradient_store._eval_hashes), 8)    # 4 rounds (7-10) * 2 clients

    def test_leakage_failure(self):
        grad = torch.randn(100)
        
        # train
        self.tracker.gradient_store.register_train_hash(grad)
        
        # eval (same gradient) - should raise AssertionError
        with self.assertRaises(AssertionError):
            self.tracker.gradient_store.register_eval_hash(grad)
            
        # Bypass protection to just manually verify the disjointness logic if they were to get in
        grad_hash = self.tracker.gradient_store._train_hashes.copy().pop()
        self.tracker.gradient_store._eval_hashes.add(grad_hash)
        self.assertFalse(self.tracker.gradient_store._train_hashes.isdisjoint(self.tracker.gradient_store._eval_hashes))

    def test_prevent_eval_store(self):
        # Attempt to store gradient from an evaluation round
        update = [ClientUpdate(
            client_id=0, 
            defended_gradients=torch.randn(100), 
            n_samples=100,
            weight_delta=OrderedDict(),
            raw_gradients=torch.randn(100),
            local_loss=0.5
        )]
        self.attack.collect(10, update) # Since collect_rounds=5
        
        self.assertEqual(len(self.tracker.gradient_store._train_hashes), 0)

    def test_adversarial_leakage_rejection(self):
        # Try forcing identical eval data directly into the collect store tracking
        bad_tensor = torch.randn(100)
        self.tracker.gradient_store.register_eval_hash(bad_tensor)
        
        with self.assertRaises(AssertionError):
            self.tracker.gradient_store.store(1, 0, bad_tensor)

if __name__ == '__main__':
    unittest.main()
