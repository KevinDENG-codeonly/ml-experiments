#!/usr/bin/env python
"""
Script to check if all required dependencies are installed and set up the environment.
"""

import sys
import subprocess
import os
from pathlib import Path


def check_dependency(package_name):
    """Check if a package is installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def check_environment():
    """Check if all required dependencies are installed."""
    # Read requirements.txt
    requirements_path = Path("requirements.txt")
    if not requirements_path.exists():
        print("requirements.txt not found.")
        return False
    
    with open(requirements_path, "r") as f:
        requirements = f.readlines()
    
    # Parse requirements (simplified parsing)
    packages = []
    for req in requirements:
        req = req.strip()
        if not req or req.startswith("#"):
            continue
        
        # Extract package name (ignoring version)
        package = req.split(">=")[0].split("==")[0].strip()
        packages.append(package)
    
    # Check dependencies
    missing_packages = []
    for package in packages:
        if not check_dependency(package):
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing dependencies:")
        for package in missing_packages:
            print(f"  - {package}")
        
        # Ask to install missing packages
        install = input("Do you want to install missing packages? (y/n): ")
        if install.lower() == "y":
            print("Installing missing packages...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("Dependencies installed successfully.")
        else:
            print("Please install the missing packages manually.")
            return False
    else:
        print("All dependencies are installed.")
    
    return True


def create_directories():
    """Create necessary directories."""
    directories = [
        "data",
        "outputs/logs",
        "outputs/models",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def main():
    """Main function."""
    print("Checking environment...")
    if check_environment():
        print("Environment is ready.")
        create_directories()
        print("Setup completed successfully.")
    else:
        print("Environment setup failed.")


if __name__ == "__main__":
    main() 