#!/usr/bin/env python3
"""
Environment test script for general capabilities evaluation setup.
Tests all required dependencies and GPU availability.
"""

import sys
import subprocess

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def test_python_version():
    """Test Python version"""
    print_section("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 10:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python 3.10+ required")
        return False

def test_imports():
    """Test importing required packages"""
    print_section("Package Imports")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("deepeval", "DeepEval"),
        ("accelerate", "Accelerate"),
        ("sentencepiece", "SentencePiece"),
        ("tiktoken", "TikToken"),
        ("einops", "Einops"),
        ("safetensors", "SafeTensors"),
    ]
    
    all_passed = True
    for package, name in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"✓ {name}: {version}")
        except ImportError as e:
            print(f"✗ {name}: Import failed - {e}")
            all_passed = False
    
    return all_passed

def test_torch_cuda():
    """Test PyTorch CUDA availability"""
    print_section("PyTorch CUDA")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print("✓ CUDA is properly configured")
            return True
        else:
            print("⚠ CUDA not available (CPU-only mode)")
            return True  # Not a failure, just a warning
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False

def test_transformers_models():
    """Test loading a small model"""
    print_section("Transformers Model Loading")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print("Testing with GPT2 (small test model)...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Test inference
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5)
        result = tokenizer.decode(outputs[0])
        
        print(f"Test generation: {result[:50]}...")
        print("✓ Model loading and inference works")
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_deepeval():
    """Test DeepEval functionality"""
    print_section("DeepEval Benchmark")
    
    try:
        from deepeval.benchmarks import HellaSwag
        from deepeval.models.base_model import DeepEvalBaseLLM
        
        # Create a simple test model
        class TestModel(DeepEvalBaseLLM):
            def __init__(self):
                pass
            
            def load_model(self):
                return None
            
            def generate(self, prompt: str) -> str:
                return "test"
            
            async def a_generate(self, prompt: str) -> str:
                return self.generate(prompt)
            
            def get_model_name(self):
                return "test_model"
        
        # Test initialization (don't run full benchmark)
        benchmark = HellaSwag(n_problems=1)
        print(f"✓ HellaSwag benchmark initialized")
        return True
    except Exception as e:
        print(f"✗ DeepEval test failed: {e}")
        return False

def test_environment_isolation():
    """Check if environment is properly isolated"""
    print_section("Environment Isolation")
    
    try:
        # Check conda environment
        conda_env = subprocess.run(
            ["conda", "info", "--envs"],
            capture_output=True,
            text=True,
            check=True
        )
        
        active_env = None
        for line in conda_env.stdout.split('\n'):
            if '*' in line:
                active_env = line.split()[0]
                break
        
        if active_env:
            print(f"Active conda environment: {active_env}")
            if active_env == "hellaswag_eval":
                print("✓ Using dedicated environment")
            else:
                print(f"⚠ Not in 'hellaswag_eval' environment (in '{active_env}')")
        else:
            print("⚠ Could not determine active environment")
        
        return True
    except Exception as e:
        print(f"⚠ Could not check conda environment: {e}")
        return True  # Not critical

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("  HELLASWAG ENVIRONMENT TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Python Version", test_python_version),
        ("Package Imports", test_imports),
        ("PyTorch CUDA", test_torch_cuda),
        ("Transformers", test_transformers_models),
        ("DeepEval", test_deepeval),
        ("Environment Isolation", test_environment_isolation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ Unexpected error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print_section("Test Summary")
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n✓ Environment is ready for HellaSwag evaluation!")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())