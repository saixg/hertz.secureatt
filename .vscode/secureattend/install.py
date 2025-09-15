#!/usr/bin/env python3
"""
Installation script for SecureAttend Backend
Run this script to install all required dependencies
"""

import subprocess
import sys
import os

def run_pip_install(package, description=""):
    """Install a package using pip"""
    print(f"Installing {description or package}...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
        print(f"✓ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def main():
    """Main installation function"""
    print("=" * 60)
    print("SecureAttend Backend - Dependency Installation")
    print("=" * 60)
    
    # Core FastAPI dependencies
    core_packages = [
        ("fastapi==0.104.1", "FastAPI web framework"),
        ("uvicorn[standard]==0.24.0", "ASGI server"),
        ("pydantic==2.5.0", "Data validation"),
        ("sqlmodel==0.0.14", "Database ORM"),
        ("python-multipart==0.0.6", "File upload support"),
        ("websockets==12.0", "WebSocket support"),
        ("Pillow==10.1.0", "Image processing"),
        ("aiofiles==23.2.1", "Async file operations")
    ]
    
    # ML dependencies
    ml_packages = [
        ("opencv-python==4.8.1.78", "Computer vision"),
        ("numpy==1.24.3", "Numerical computing"),
        ("face-recognition==1.3.0", "Face recognition library")
    ]
    
    # Optional dependencies
    optional_packages = [
        ("alembic==1.12.1", "Database migrations"),
        ("python-jose[cryptography]==3.3.0", "JWT tokens"),
        ("passlib[bcrypt]==1.7.4", "Password hashing")
    ]
    
    print("\nStep 1: Upgrading pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    print("\nStep 2: Installing core FastAPI dependencies...")
    failed_core = []
    for package, desc in core_packages:
        if not run_pip_install(package, desc):
            failed_core.append(package)
    
    print("\nStep 3: Installing ML dependencies...")
    failed_ml = []
    for package, desc in ml_packages:
        if not run_pip_install(package, desc):
            failed_ml.append(package)
    
    print("\nStep 4: Installing optional dependencies...")
    failed_optional = []
    for package, desc in optional_packages:
        if not run_pip_install(package, desc):
            failed_optional.append(package)
    
    print("\n" + "=" * 60)
    print("Installation Summary")
    print("=" * 60)
    
    if not failed_core and not failed_ml:
        print("✓ All core dependencies installed successfully!")
        print("\nNext steps:")
        print("1. Create 'known_faces' folder")
        print("2. Add student photos: hasini.jpg, anji.jpg, venkat.jpg, sai_reddy.jpg")
        print("3. Run: python main.py")
        print("\nYour backend will be available at: http://localhost:8000")
    else:
        print("⚠ Some packages failed to install:")
        if failed_core:
            print("Core packages:", failed_core)
        if failed_ml:
            print("ML packages:", failed_ml)
        if failed_optional:
            print("Optional packages:", failed_optional)
        
        print("\nTroubleshooting:")
        print("- Make sure you have Python 3.8+ installed")
        print("- On Windows: Install Visual Studio Build Tools")
        print("- On Ubuntu: sudo apt-get install cmake libopenblas-dev liblapack-dev")
        print("- On macOS: brew install cmake")

if __name__ == "__main__":
    main()