#!/usr/bin/env python3
"""
Quick test script to demonstrate success rate tracking functionality.
This runs a short PPO training and verifies all features work.
"""

import subprocess
import os
import sys

def main():
    print("=" * 70)
    print("  SUCCESS RATE TRACKING - QUICK TEST")
    print("=" * 70)
    print()
    print("This will run a short PPO training to verify success rate tracking.")
    print("Environment: MiniGrid-Empty-5x5-v0 (easy environment)")
    print("Episodes: 200 (should take ~2-3 minutes)")
    print()

    # Test parameters
    model_name = "test_success_rate_demo"

    # Clean up any existing test results
    test_dir = os.path.join("storage", model_name)
    if os.path.exists(test_dir):
        print(f"⚠️  Removing existing test directory: {test_dir}")
        import shutil
        shutil.rmtree(test_dir)
        print()

    print("Starting training...")
    print("-" * 70)

    # Run training
    cmd = [
        sys.executable, "scripts/train.py",
        "--algo", "ppo",
        "--env", "MiniGrid-Empty-5x5-v0",
        "--model", model_name,
        "--episodes", "200",
        "--procs", "4",
        "--frames-per-proc", "128",
        "--epochs", "4",
        "--batch-size", "128",
        "--save-interval", "10",
        "--log-interval", "5"
    ]

    try:
        result = subprocess.run(cmd, check=True)

        print()
        print("-" * 70)
        print("✅ Training completed successfully!")
        print()

        # Check outputs
        print("Checking generated files...")
        print()

        files_to_check = [
            ("status.pt", "Checkpoint file"),
            ("log.txt", "Text log"),
            ("log.csv", "CSV log"),
            ("success_rate_plot.png", "Success rate plot")
        ]

        all_found = True
        for filename, description in files_to_check:
            filepath = os.path.join(test_dir, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"  ✅ {description:25s} - {filename:25s} ({size:,} bytes)")
            else:
                print(f"  ❌ {description:25s} - {filename:25s} (NOT FOUND)")
                all_found = False

        print()
        print("=" * 70)

        if all_found:
            print("✅ ALL FEATURES WORKING CORRECTLY!")
            print()
            print("To view the success rate plot:")
            print(f"  → {os.path.join(test_dir, 'success_rate_plot.png')}")
            print()
            print("To view detailed logs:")
            print(f"  → {os.path.join(test_dir, 'log.txt')}")
            print(f"  → {os.path.join(test_dir, 'log.csv')}")
        else:
            print("⚠️  Some files are missing. Check the training logs for errors.")

        print("=" * 70)

    except subprocess.CalledProcessError as e:
        print()
        print("=" * 70)
        print(f"❌ Training failed with exit code {e.returncode}")
        print("=" * 70)
        return 1
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("⚠️  Training interrupted by user")
        print("=" * 70)
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
