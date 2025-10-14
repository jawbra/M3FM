#!/usr/bin/env python3
"""
Lung Cancer Risk Inference Script - Simplified Version
For parallel execution, use: run_inference_parallel.py

This is a simplified version that runs both lungs sequentially and combines results
For more control and parallel execution, use run_inference_parallel.py
"""

import sys
import subprocess
import argparse

def main():
    """Main function - runs both lungs and combines by default"""
    
    parser = argparse.ArgumentParser(description='Run lung cancer risk inference (simplified)')
    parser.add_argument('lung_side', 
                       nargs='?',  # Optional argument
                       choices=['left', 'right', 'both'], 
                       default='both',
                       help='Which lung(s) to process (default: both)')
    
    args = parser.parse_args()
    
    print("🚀 Starting lung cancer risk inference...")
    print(f"📋 Processing: {args.lung_side}")
    
    if args.lung_side == 'both':
        print("📝 Note: For parallel execution in separate terminals, use:")
        print("   Terminal 1: python run_inference_parallel.py left")  
        print("   Terminal 2: python run_inference_parallel.py right")
        print("   Then combine: python merge_outputs.py")
        print()
    
    # Call the parallel script with appropriate arguments
    cmd = ['python', 'run_inference_parallel.py', args.lung_side]
    if args.lung_side == 'both':
        cmd.append('--combine')
    
    try:
        # Run the parallel script
        result = subprocess.run(cmd, check=True)
        print("✅ Inference completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Inference failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Inference interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()