import torch
import torch.optim.lr_scheduler as lr_scheduler


# On server side
class FederatedScheduler:
    def __init__(self, initial_lr=0.01, total_rounds=100, scheduler_type="cosine", scheduler_params=None):
        self.current_round = 0
        self.total_rounds = total_rounds
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.last_processed_round = 0

        # Create a mock optimizer just for the scheduler
        mock_params = [torch.nn.Parameter(torch.zeros(1))]
        self.mock_optimizer = torch.optim.SGD(mock_params, lr=initial_lr)

        # Default scheduler parameters if none provided
        if scheduler_params is None:
            scheduler_params = {}

        self.scheduler_type = scheduler_type.lower()
        self.milestones = self._normalize_milestones(
            scheduler_params.get("milestones", [])
        )
        self.milestone_gamma = float(scheduler_params.get("gamma", 0.1))
        self.last_milestone = self.milestones[-1] if self.milestones else None
        self.cosine_start_round = None

        if any(milestone > self.total_rounds for milestone in self.milestones):
            raise ValueError("LR milestones cannot exceed total_rounds")

        # Choose scheduler based on string argument
        self.scheduler = self._create_scheduler(self.scheduler_type, scheduler_params)

    @staticmethod
    def _normalize_milestones(milestones):
        """Return sorted, unique, positive integer milestone rounds."""
        if milestones is None:
            return []
        if isinstance(milestones, str):
            milestones = [
                value.strip()
                for value in milestones.split(",")
                if value.strip()
            ]

        normalized = sorted({int(milestone) for milestone in milestones})
        if any(milestone <= 0 for milestone in normalized):
            raise ValueError("LR milestones must be positive round numbers")
        return normalized

    def _create_scheduler(self, scheduler_type, params):
        """Create and return the specified learning rate scheduler"""
        scheduler_type = scheduler_type.lower()

        if scheduler_type == "milestone_then_cosine":
            if not self.milestones:
                raise ValueError(
                    "milestone_then_cosine requires at least one LR milestone"
                )
            # The cosine scheduler is initialized at the final milestone, after
            # applying the last factor-of-ten reduction.
            return None
        elif scheduler_type == "cosine":
            return lr_scheduler.CosineAnnealingLR(
                self.mock_optimizer,
                T_max=params.get("T_max", self.total_rounds),
                eta_min=params.get("eta_min", 0)
            )
        elif scheduler_type == "step":
            return lr_scheduler.StepLR(
                self.mock_optimizer,
                step_size=params.get("step_size", self.total_rounds // 4),
                gamma=params.get("gamma", 0.1)
            )
        elif scheduler_type == "multistep":
            return lr_scheduler.MultiStepLR(
                self.mock_optimizer,
                milestones=params.get("milestones", [self.total_rounds // 3, self.total_rounds // 3 * 2]),
                gamma=params.get("gamma", 0.1)
            )
        elif scheduler_type == "exponential":
            return lr_scheduler.ExponentialLR(
                self.mock_optimizer,
                gamma=params.get("gamma", 0.95)
            )
        elif scheduler_type == "plateau":
            return lr_scheduler.ReduceLROnPlateau(
                self.mock_optimizer,
                mode=params.get("mode", "min"),
                factor=params.get("factor", 0.1),
                patience=params.get("patience", 10),
                verbose=params.get("verbose", False)
            )
        elif scheduler_type == "onecycle":
            return lr_scheduler.OneCycleLR(
                self.mock_optimizer,
                max_lr=params.get("max_lr", self.initial_lr * 10),
                total_steps=params.get("total_steps", self.total_rounds),
                pct_start=params.get("pct_start", 0.3)
            )
        elif scheduler_type == "cyclic":
            return lr_scheduler.CyclicLR(
                self.mock_optimizer,
                base_lr=params.get("base_lr", self.initial_lr / 10),
                max_lr=params.get("max_lr", self.initial_lr),
                step_size_up=params.get("step_size_up", self.total_rounds // 6),
                mode=params.get("mode", "triangular")
            )
        elif scheduler_type == "constant":
            # No scheduling, maintain constant learning rate
            return lr_scheduler.LambdaLR(self.mock_optimizer, lambda epoch: 1.0)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def _set_learning_rate(self, learning_rate):
        """Update both the exposed LR and the scheduler's mock optimizer."""
        self.current_lr = float(learning_rate)
        for param_group in self.mock_optimizer.param_groups:
            param_group["lr"] = self.current_lr

    def _start_cosine_after_milestones(self, server_round):
        """Restart cosine decay from the fully milestone-reduced LR."""
        remaining_rounds = self.total_rounds - server_round
        self.cosine_start_round = server_round
        if remaining_rounds <= 0:
            self.scheduler = None
            return

        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.mock_optimizer,
            T_max=remaining_rounds,
            eta_min=0,
        )

    def _process_milestone_then_cosine_round(self, server_round):
        """Apply an exact-round milestone or advance the restarted cosine."""
        if server_round in self.milestones:
            self._set_learning_rate(self.current_lr * self.milestone_gamma)
            if server_round == self.last_milestone:
                self._start_cosine_after_milestones(server_round)
            return

        if (
            self.cosine_start_round is not None
            and server_round > self.cosine_start_round
            and self.scheduler is not None
        ):
            self.scheduler.step()
            self._set_learning_rate(self.scheduler.get_last_lr()[0])

    def _process_standard_round(self, server_round):
        """Advance an existing scheduler before rounds after round one."""
        if server_round == 1:
            return
        if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            raise ValueError(
                "Metrics are required to advance ReduceLROnPlateau by server round"
            )
        self.scheduler.step()
        self._set_learning_rate(self.scheduler.get_last_lr()[0])

    def get_learning_rate_for_round(self, server_round=None):
        """Return the learning rate for an exact server round.

        Repeated requests for the same round are idempotent. If rounds are
        skipped, each intervening round is processed so exact milestones are
        still applied once.
        """
        if server_round is None:
            return self.current_lr

        server_round = int(server_round)
        if server_round <= 0:
            raise ValueError("server_round must be positive")
        if server_round < self.last_processed_round:
            raise ValueError("Cannot request a learning rate for an earlier round")
        if server_round == self.last_processed_round:
            return self.current_lr

        for round_number in range(self.last_processed_round + 1, server_round + 1):
            if self.scheduler_type == "milestone_then_cosine":
                self._process_milestone_then_cosine_round(round_number)
            else:
                self._process_standard_round(round_number)
            self.current_round = round_number

        self.last_processed_round = server_round
        return self.current_lr

    def update_after_round(self, global_model=None, metrics=None):
        """Update the scheduler after aggregating client updates"""
        self.current_round += 1

        # Handle ReduceLROnPlateau differently since it requires a metric
        if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            if metrics is None:
                raise ValueError("Metrics required for ReduceLROnPlateau scheduler")
            self.scheduler.step(metrics)
        else:
            # Step the scheduler
            self.scheduler.step()

        # Get the new learning rate
        if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            # For ReduceLROnPlateau, we need to get the lr from the optimizer directly
            self.current_lr = self.mock_optimizer.param_groups[0]['lr']
        else:
            self.current_lr = self.scheduler.get_last_lr()[0]

        return self.current_lr

    def get_scheduler_info(self):
        """Return information about the current scheduler"""
        scheduler_type = (
            "MilestoneThenCosine"
            if self.scheduler_type == "milestone_then_cosine"
            else type(self.scheduler).__name__
        )
        return {
            "type": scheduler_type,
            "current_lr": self.current_lr,
            "round": self.current_round,
            "total_rounds": self.total_rounds
        }
