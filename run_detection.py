#!/usr/bin/env python3
"""
Bread Anomaly Detection System - Main Entry Point
=====================================================

This script provides an easy-to-use interface for running the bread detection system.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_model():
    """Check if the improved model exists"""
    model_path = Path("models/padim_bread_improved.pth")
    return model_path.exists()

def check_data():
    """Check if training data exists"""
    data_path = Path("data/bread_dataset/train/normal")
    return data_path.exists() and len(list(data_path.glob("*.jpg"))) > 0

def run_training():
    """Run the training script"""
    print("🧠 Starting model training...")
    try:
        result = subprocess.run([sys.executable, "src/improved_train_padim.py"], 
                              check=True, capture_output=True, text=True)
        print("✅ Training completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Training failed!")
        print(e.stderr)
        return False

def run_detection():
    """Run the detection script"""
    print("🎯 Starting bread detection system...")
    try:
        subprocess.run([sys.executable, "src/improved_padim_deploy.py"], check=True)
    except subprocess.CalledProcessError as e:
        print("❌ Detection failed!")
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\n👋 Detection stopped by user")

def run_data_organizer():
    """Run the data organization utility"""
    print("🗃️ Starting data organizer...")
    try:
        subprocess.run([sys.executable, "utils/data_organizer.py"], check=True)
    except subprocess.CalledProcessError as e:
        print("❌ Data organizer failed!")
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Bread Anomaly Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_detection.py                    # Run detection (default)
  python run_detection.py --train           # Train new model
  python run_detection.py --data            # Organize data
  python run_detection.py --setup           # Complete setup
        """
    )
    
    parser.add_argument("--train", action="store_true", 
                       help="Train a new model")
    parser.add_argument("--data", action="store_true", 
                       help="Run data organizer")
    parser.add_argument("--setup", action="store_true", 
                       help="Complete setup (data + training)")
    parser.add_argument("--force", action="store_true", 
                       help="Force retrain even if model exists")
    
    args = parser.parse_args()
    
    print("🍞 Bread Anomaly Detection System")
    print("=" * 40)
    
    # Check system status
    has_model = check_model()
    has_data = check_data()
    
    print(f"📊 Training data: {'✅ Found' if has_data else '❌ Missing'}")
    print(f"🧠 Trained model: {'✅ Found' if has_model else '❌ Missing'}")
    print()
    
    if args.setup:
        print("🔧 Running complete setup...")
        if not has_data:
            print("Please organize your data first:")
            run_data_organizer()
            return
        
        if not has_model or args.force:
            if not run_training():
                return
        
        run_detection()
        
    elif args.data:
        run_data_organizer()
        
    elif args.train:
        if not has_data:
            print("❌ No training data found!")
            print("Please run: python run_detection.py --data")
            return
        
        if has_model and not args.force:
            response = input("Model already exists. Retrain? (y/N): ")
            if response.lower() != 'y':
                print("Skipping training.")
                return
        
        run_training()
        
    else:
        # Default: run detection
        if not has_model:
            print("❌ No trained model found!")
            print("Please run: python run_detection.py --setup")
            return
        
        run_detection()

if __name__ == "__main__":
    main()