# verify_deepspeed_moe.py
import sys
import torch
import subprocess

def check_basic_installation():
    """Check basic DeepSpeed installation"""
    print("=== Basic Installation Checks ===")
    
    try:
        import deepspeed
        print(f"✓ DeepSpeed version: {deepspeed.__version__}")
    except ImportError:
        print("✗ DeepSpeed not installed")
        return False
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.version.cuda}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
    else:
        print("✗ CUDA not available")
        return False
    
    return True

def check_moe_support():
    """Check MoE specific components"""
    print("\n=== MoE Component Checks ===")
    
    try:
        from deepspeed.moe.layer import MoE
        print("✓ DeepSpeed MoE module importable")
    except ImportError as e:
        print(f"✗ MoE import failed: {e}")
        return False
    
    try:
        from deepspeed.moe.experts import Experts
        print("✓ MoE Experts module available")
    except ImportError:
        print("✗ MoE Experts module not available")
    
    try:
        from deepspeed.moe.sharded_moe import TopKGate
        print("✓ TopKGate available")
    except ImportError:
        print("✗ TopKGate not available")
    
    return True

def check_tutel():
    """Check Tutel installation for high-performance MoE"""
    print("\n=== Tutel Installation Check ===")
    
    try:
        import tutel
        print(f"✓ Tutel installed")
        
        # Check if tutel MoE is available
        from tutel import moe as tutel_moe
        print("✓ Tutel MoE module available")
        return True
    except ImportError:
        print("✗ Tutel not installed (optional but recommended for performance)")
        return False

def test_simple_moe_creation():
    """Test creating a simple MoE layer"""
    print("\n=== MoE Layer Creation Test ===")
    
    try:
        from deepspeed.moe.layer import MoE
        import torch.nn as nn
        
        # Simple expert class
        class SimpleExpert(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.linear = nn.Linear(hidden_size, hidden_size)
            
            def forward(self, x):
                return self.linear(x)
        
        # Create MoE layer
        hidden_size = 768
        num_experts = 8
        k = 2
        
        moe = MoE(
            hidden_size=hidden_size,
            expert=SimpleExpert(hidden_size),
            num_experts=num_experts,
            k=k,
            capacity_factor=1.25,
            eval_capacity_factor=2.0,
            min_capacity=4,
            use_residual=False,
        )
        
        print(f"✓ MoE layer created successfully")
        print(f"  - Hidden size: {hidden_size}")
        print(f"  - Experts: {num_experts}")
        print(f"  - Top-k: {k}")
        
        return True
    except Exception as e:
        print(f"✗ MoE layer creation failed: {e}")
        return False

def test_moe_forward_pass():
    """Test a forward pass through MoE"""
    print("\n=== MoE Forward Pass Test ===")
    
    try:
        from deepspeed.moe.layer import MoE
        import torch.nn as nn
        import torch
        
        class SimpleExpert(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.linear = nn.Linear(hidden_size, hidden_size)
                self.activation = nn.ReLU()
            
            def forward(self, x):
                return self.activation(self.linear(x))
        
        hidden_size = 512
        batch_size = 4
        seq_len = 32
        
        moe = MoE(
            hidden_size=hidden_size,
            expert=SimpleExpert(hidden_size),
            num_experts=8,
            k=2,
            capacity_factor=1.25,
        )
        
        # Create test input
        x = torch.randn(batch_size * seq_len, hidden_size)
        
        if torch.cuda.is_available():
            moe = moe.cuda()
            x = x.cuda()
            print("✓ Using CUDA for test")
        
        # Forward pass
        output, gate_loss, metadata = moe(x)
        
        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Gate loss: {gate_loss.item():.6f}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def check_deepspeed_ops():
    """Check if DeepSpeed ops are properly compiled"""
    print("\n=== DeepSpeed Ops Check ===")
    
    try:
        import deepspeed.ops
        print("✓ DeepSpeed ops module available")
        
        # Check specific MoE ops
        try:
            from deepspeed.ops.sparse_attention import SparseSelfAttention
            print("✓ Sparse attention ops available")
        except ImportError:
            print("! Sparse attention ops not available (may not be needed)")
        
        return True
    except Exception as e:
        print(f"! DeepSpeed ops check: {e}")
        return False

def run_deepspeed_info():
    """Run deepspeed environment info"""
    print("\n=== DeepSpeed Environment Info ===")
    
    try:
        result = subprocess.run(['ds_report'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ DeepSpeed environment report:")
            print(result.stdout)
        else:
            print("! ds_report command failed")
    except FileNotFoundError:
        print("! ds_report command not found")
    except Exception as e:
        print(f"! Error running ds_report: {e}")

def main():
    print("DeepSpeed MoE Installation Verification")
    print("=" * 50)
    
    all_passed = True
    
    # Run all checks
    all_passed &= check_basic_installation()
    all_passed &= check_moe_support()
    check_tutel()  # Optional
    check_deepspeed_ops()  # Optional
    all_passed &= test_simple_moe_creation()
    all_passed &= test_moe_forward_pass()
    
    # Environment info
    run_deepspeed_info()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All critical tests passed! DeepSpeed MoE is ready to use.")
    else:
        print("✗ Some tests failed. Check installation.")
        
    print("\nInstallation commands if needed:")
    print("pip install deepspeed[moe]")
    print("pip install tutel  # Optional but recommended")

if __name__ == "__main__":
    main()