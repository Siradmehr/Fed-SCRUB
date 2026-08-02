import unittest

from src.utils.lr_scheduler import FederatedScheduler


class FederatedSchedulerTests(unittest.TestCase):
    def make_hybrid_scheduler(self):
        return FederatedScheduler(
            initial_lr=0.2,
            total_rounds=20,
            scheduler_type="milestone_then_cosine",
            scheduler_params={
                "milestones": [5, 10],
                "gamma": 0.1,
            },
        )

    def test_exact_milestones_then_restart_cosine(self):
        scheduler = self.make_hybrid_scheduler()

        learning_rates = {
            server_round: scheduler.get_learning_rate_for_round(server_round)
            for server_round in range(1, 21)
        }

        self.assertAlmostEqual(learning_rates[1], 0.2)
        self.assertAlmostEqual(learning_rates[4], 0.2)
        self.assertAlmostEqual(learning_rates[5], 0.02)
        self.assertAlmostEqual(learning_rates[9], 0.02)
        self.assertAlmostEqual(learning_rates[10], 0.002)
        self.assertLess(learning_rates[11], 0.002)
        self.assertAlmostEqual(learning_rates[20], 0.0)

    def test_repeated_round_request_does_not_apply_milestone_twice(self):
        scheduler = self.make_hybrid_scheduler()

        first_value = scheduler.get_learning_rate_for_round(5)
        repeated_value = scheduler.get_learning_rate_for_round(5)

        self.assertAlmostEqual(first_value, 0.02)
        self.assertAlmostEqual(repeated_value, 0.02)

    def test_string_milestones_are_normalized(self):
        scheduler = FederatedScheduler(
            initial_lr=0.2,
            total_rounds=20,
            scheduler_type="milestone_then_cosine",
            scheduler_params={"milestones": "10,5,5"},
        )

        self.assertEqual(scheduler.milestones, [5, 10])


if __name__ == "__main__":
    unittest.main()
