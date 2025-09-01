# First, let's test basic wandb connectivity
import wandb
import yaml


def test_basic_connection():
    """Test if basic wandb connection works"""
    print("Testing basic wandb connection...")
    try:
        run = wandb.init(project="test-connection", mode="online")
        wandb.log({"test_metric": 1.0})
        wandb.finish()
        print("✅ Basic connection works!")
        return True
    except Exception as e:
        print(f"❌ Basic connection failed: {e}")
        return False


def test_sweep_without_entity():
    """Test sweep creation without entity"""
    print("Testing sweep without entity...")
    try:
        # Simple sweep config
        sweep_config = {
            'method': 'grid',
            'metric': {'name': 'val_loss', 'goal': 'minimize'},
            'parameters': {
                'LR': {'values': [0.01, 0.1]}
            }
        }

        # Create sweep WITHOUT entity
        sweep_id = wandb.sweep(sweep_config, project="fed-scrub-test")
        print(f"✅ Sweep created successfully! ID: {sweep_id}")
        return True
    except Exception as e:
        print(f"❌ Sweep creation failed: {e}")
        return False


def test_original_config():
    """Test with your original sweep config but no entity"""
    print("Testing with original config (no entity)...")
    try:
        with open("wandb_sweep.yaml") as f:
            sweep_config = yaml.safe_load(f)

        print("Loaded sweep config:", sweep_config)

        # Create sweep WITHOUT entity
        sweep_id = wandb.sweep(sweep_config, project="fed-scrub")
        print(f"✅ Original config works! ID: {sweep_id}")
        return True
    except Exception as e:
        print(f"❌ Original config failed: {e}")
        return False


if __name__ == "__main__":
    print("=== WandB Debug Tests ===\n")

    # Run tests in order
    if test_basic_connection():
        if test_sweep_without_entity():
            test_original_config()
        else:
            print("Skipping original config test due to sweep failure")
    else:
        print("Skipping all other tests due to basic connection failure")

    print("\n=== Debug Complete ===")