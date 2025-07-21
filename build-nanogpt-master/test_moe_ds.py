import torch
import torch.nn as nn
import deepspeed
from deepspeed.moe.layer import MoE

# --------------------------
# 1. Dummy dataset generator
# --------------------------
def random_data(batch_size=4, hidden_dim=16):
    x = torch.randn(batch_size, hidden_dim)
    y = torch.randn(batch_size, hidden_dim)
    return x, y

# --------------------------
# 2. Define a simple MoE model
# --------------------------
class TinyMoEModel(nn.Module):
    def __init__(self, hidden_dim=16, num_experts=4, k=1):
        super().__init__()
        # A small input projection
        self.input_layer = nn.Linear(hidden_dim, hidden_dim)

        # Mixture of Experts layer
        self.moe = MoE(
            hidden_size=hidden_dim,
            expert=self._make_expert(hidden_dim),
            num_experts=num_experts,
            ep_size=1,         # expert parallelism = 1 for this test
            k=k,               # top-k experts per token
            use_residual=True  # optional residual connection
        )

        # A small output projection
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def _make_expert(self, hidden_dim):
        """Each expert is a simple feedforward layer"""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x, _ = self.moe(x)  # MoE returns (output, auxiliary_loss)
        x = self.output_layer(x)
        return x

# --------------------------
# 3. Setup model & DeepSpeed
# --------------------------
def main():
    hidden_dim = 16
    batch_size = 4

    model = TinyMoEModel(hidden_dim=hidden_dim, num_experts=4, k=1)

    # DeepSpeed config path
    ds_config = {
    "train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.001
        }
    },
    "fp16": {
        "enabled": False
    },
    "moe": {
        "enabled": True,
        "moe_param_group": True
    }
}

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    # Dummy data
    x, y = random_data(batch_size=batch_size, hidden_dim=hidden_dim)
    x, y = x.to(model_engine.device), y.to(model_engine.device)

    # Forward
    outputs = model_engine(x)
    loss_fn = nn.MSELoss()
    loss = loss_fn(outputs, y)

    # Backward + step
    model_engine.backward(loss)
    model_engine.step()

    print("✅ DeepSpeed MoE single training step completed successfully!")

if __name__ == "__main__":
    main()