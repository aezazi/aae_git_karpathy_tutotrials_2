#!/usr/bin/env python3

import torch
import torch.nn as nn
import deepspeed
import os

def initialize_deepspeed_distributed():
    """Initialize DeepSpeed's distributed backend for single GPU"""
    print("=== Initializing DeepSpeed Distributed Backend ===")
    
    # Set environment variables for single-process distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0' 
    os.environ['WORLD_SIZE'] = '1'
    
    try:
        # Initialize distributed backend
        deepspeed.init_distributed()
        print("✓ DeepSpeed distributed backend initialized")
        return True
    except Exception as e:
        print(f"! Could not initialize distributed backend: {e}")
        try:
            # Alternative: Initialize torch distributed directly
            torch.distributed.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                init_method='env://',
                world_size=1,
                rank=0
            )
            print("✓ PyTorch distributed backend initialized")
            return True
        except Exception as e2:
            print(f"! Could not initialize any distributed backend: {e2}")
            return False

class SimpleExpert(nn.Module):
    """Simple expert for testing"""
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 2)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))

def test_moe_complete():
    """Complete MoE test with proper initialization"""
    print("DeepSpeed MoE Complete Test")
    print("=" * 40)
    
    # Initialize distributed backend
    if not initialize_deepspeed_distributed():
        print("❌ Cannot run MoE without distributed backend")
        print("💡 Try running with: python -m torch.distributed.launch --nproc_per_node=1 test_moe.py")
        return False
    
    print("\n=== Creating MoE Layer ===")
    
    try:
        from deepspeed.moe.layer import MoE
        
        # MoE configuration
        hidden_size = 512
        num_experts = 4
        k = 2
        
        # Create expert
        expert = SimpleExpert(hidden_size)
        
        # Create MoE layer
        moe_layer = MoE(
            hidden_size=hidden_size,
            expert=expert,
            num_experts=num_experts,
            k=k,
            capacity_factor=1.25,
            eval_capacity_factor=2.0,
            min_capacity=4,
            use_residual=False,
            drop_tokens=True,
            use_rts=True,
            use_tutel=False  # Set to True if you install tutel
        )
        
        print(f"✓ MoE layer created successfully")
        print(f"  - Hidden size: {hidden_size}")
        print(f"  - Number of experts: {num_experts}")
        print(f"  - Top-k: {k}")
        print(f"  - Use Tutel: False")
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        moe_layer = moe_layer.to(device)
        print(f"✓ Moved to device: {device}")
        
    except Exception as e:
        print(f"❌ MoE layer creation failed: {e}")
        return False
    
    print("\n=== Testing Forward Pass ===")
    
    try:
        # Create test input
        batch_size = 2
        seq_len = 16
        input_tensor = torch.randn(batch_size * seq_len, hidden_size, device=device)
        
        print(f"Input shape: {input_tensor.shape}")
        
        # Set to training mode
        moe_layer.train()
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=False):  # Disable autocast for stability
            output, gate_loss, metadata = moe_layer(input_tensor)
        
        print(f"✓ Forward pass successful!")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Gate loss: {gate_loss.item():.6f}")
        print(f"  - Metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else 'Not a dict'}")
        
        # Test backward pass
        print("\n=== Testing Backward Pass ===")
        loss = output.mean() + gate_loss
        loss.backward()
        print("✓ Backward pass successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward/backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_tutel():
    """Test MoE with Tutel if available"""
    print("\n=== Testing with Tutel (if available) ===")
    
    try:
        import tutel
        print(f"✓ Tutel available: {tutel.__version__}")
        
        from deepspeed.moe.layer import MoE
        
        hidden_size = 256
        expert = SimpleExpert(hidden_size)
        
        moe_layer = MoE(
            hidden_size=hidden_size,
            expert=expert,
            num_experts=4,
            k=2,
            use_tutel=True  # Enable Tutel
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        moe_layer = moe_layer.to(device)
        
        # Test forward pass
        input_tensor = torch.randn(32, hidden_size, device=device)
        output, gate_loss, metadata = moe_layer(input_tensor)
        
        print("✓ Tutel-enabled MoE works!")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Gate loss: {gate_loss.item():.6f}")
        
        return True
        
    except ImportError:
        print("! Tutel not available - install with: pip install tutel")
        return False
    except Exception as e:
        print(f"! Tutel test failed: {e}")
        return False

def main():
    success = test_moe_complete()
    
    if success:
        print("\n🎉 SUCCESS: DeepSpeed MoE is working correctly!")
        
        # Optional: Test with Tutel
        test_with_tutel()
        
        print("\n📋 Summary:")
        print("✅ MoE layer creation: Working")
        print("✅ Forward pass: Working") 
        print("✅ Backward pass: Working")
        print("✅ Distributed backend: Working")
        
        print("\n💡 Your DeepSpeed MoE setup is ready for training!")
        print("Use: from deepspeed.moe.layer import MoE")
        
    else:
        print("\n❌ Some issues remain. Try running with torch.distributed.launch:")
        print("python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 test_moe.py")

if __name__ == "__main__":
    main()