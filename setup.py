#!/usr/bin/env python3
"""
Setup and installation script for the telecoms fraud detection ML project.
"""

import sys
import subprocess
from pathlib import Path


def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"   ✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {description} failed: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False


def check_uv_installed():
    """Check if uv is installed."""
    try:
        subprocess.run(['uv', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_uv():
    """Install uv package manager."""
    print("📦 Installing uv package manager...")
    install_cmd = "curl -LsSf https://astral.sh/uv/install.sh | sh"
    
    if run_command(install_cmd, "Installing uv"):
        print("   💡 Please restart your terminal or run: source ~/.bashrc")
        return True
    else:
        print("   ❌ Failed to install uv. Please install manually:")
        print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False


def setup_project():
    """Set up the project."""
    print("🚀 Setting up Telecoms Fraud Detection ML Project")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    
    # Check if uv is installed
    if not check_uv_installed():
        print("⚠️  uv package manager not found")
        if input("   Install uv now? (y/N): ").lower().strip() == 'y':
            if not install_uv():
                return False
        else:
            print("   Please install uv first: https://github.com/astral-sh/uv")
            return False
    else:
        print("✅ uv package manager found")
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "data/sample",
        "models",
        "reports",
        "reports/figures",
        "logs"
    ]
    
    print("\n📁 Creating project directories...")
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")
    
    # Install dependencies
    print(f"\n📦 Installing project dependencies...")
    
    # Install in development mode
    install_cmd = "uv pip install -e ."
    if not run_command(install_cmd, "Installing core dependencies"):
        print("   Trying alternative installation...")
        alt_cmd = "uv pip install -r requirements.txt"
        if not run_command(alt_cmd, "Installing from requirements.txt"):
            return False
    
    # Install development dependencies
    dev_cmd = 'uv pip install -e ".[dev]"'
    run_command(dev_cmd, "Installing development dependencies")
    
    # Test imports
    print("\n🧪 Testing package imports...")
    try:
        import pandas
        import numpy
        import sklearn
        print("   ✅ Core data science packages imported successfully")
    except ImportError as e:
        print(f"   ⚠️  Import issue: {e}")
        print("   You may need to install additional dependencies manually")
    
    # Generate sample data
    print("\n📊 Generating sample data...")
    sample_script = project_root / "scripts" / "demo.py"
    if sample_script.exists():
        demo_cmd = f"cd {project_root} && python scripts/demo.py"
        if run_command(demo_cmd, "Running demo script"):
            print("   ✅ Sample data generated and model trained")
        else:
            print("   ⚠️  Demo script failed, but setup can continue")
    
    print("\n🎉 Setup completed!")
    print("=" * 60)
    print("✅ Your telecoms fraud detection ML project is ready!")
    print("\n📋 Next steps:")
    print("   1. Check the generated sample data in data/raw/")
    print("   2. Review the trained model in models/model.pkl")
    print("   3. Run: python scripts/train_model.py --help")
    print("   4. Or use the CLI: fraud-predict --help")
    print("\n📚 Documentation:")
    print("   • README.md - Project overview and usage")
    print("   • config/config.yaml - Configuration options")
    print("   • requirements.txt - Dependencies list")
    
    return True


def main():
    """Main setup function."""
    try:
        if setup_project():
            print(f"\n🎯 Setup successful!")
            return 0
        else:
            print(f"\n❌ Setup failed!")
            return 1
    except KeyboardInterrupt:
        print(f"\n⚠️  Setup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error during setup: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
