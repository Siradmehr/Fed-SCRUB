import torch
import torch.optim.lr_scheduler as lr_scheduler


# On server side
class FederatedScheduler:
    def __init__(self, initial_lr=0.01, total_rounds=100, scheduler_type="cosine", scheduler_params=None):
        self.current_round = 0
        self.total_rounds = total_rounds
        self.initial_lr = initial_lr
        self.current_lr = initial_lr

        # Create a mock optimizer just for the scheduler
        mock_params = [torch.nn.Parameter(torch.zeros(1))]
        self.mock_optimizer = torch.optim.SGD(mock_params, lr=initial_lr)

        # Default scheduler parameters if none provided
        if scheduler_params is None:
            scheduler_params = {}

        # Choose scheduler based on string argument
        self.scheduler = self._create_scheduler(scheduler_type, scheduler_params)

    def _create_scheduler(self, scheduler_type, params):
        """Create and return the specified learning rate scheduler"""
        scheduler_type = scheduler_type.lower()

        if scheduler_type == "cosine":
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

    def get_learning_rate_for_round(self):
        """Return the current learning rate to send to clients"""
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
        scheduler_type = type(self.scheduler).__name__
        return {
            "type": scheduler_type,
            "current_lr": self.current_lr,
            "round": self.current_round,
            "total_rounds": self.total_rounds
        }