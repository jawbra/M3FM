#!/usr/bin/env python3
"""
Quick test script to verify error handling works with a small subset of data
"""

from inference_monai import Inference
import matplotlib.pyplot as plt
import numpy as np
import json
import logging
from datetime import datetime
import sys
import os

# Configure logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'test_error_handling_{timestamp}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def test_small_subset():
    """Test inference on a small subset to verify error handling"""
    
    logging.info("Starting small subset test for error handling verification")
    
    # Load the dataset configuration
    with open('data/dataset.json', 'r') as f:
        dataset_config = json.load(f)
    
    # Take only first 10 samples for quick testing
    original_data = dataset_config['testing']
    test_subset = original_data[:5]  # Use even smaller subset for faster testing
    
    logging.info(f"Testing with {len(test_subset)} samples (subset of {len(original_data)} total)")
    
    # Create a temporary dataset config for testing
    test_config = {
        'name': 'test_subset',
        'description': 'Small subset for error handling testing',
        'test': test_subset
    }
    
    # Save temporary config
    with open('test_subset_dataset.json', 'w') as f:
        json.dump(test_config, f, indent=4)
    
    try:
        # Test right lung inference
        logging.info("Testing right lung inference with error handling...")
        right_predictions, right_failed = Inference(test_subset, vis=False)
        
        logging.info(f"Right lung results: {len(right_predictions)} successful, {len(right_failed)} failed")
        if right_failed:
            logging.info(f"Failed samples: {right_failed}")
        
        # Test left lung inference  
        logging.info("Testing left lung inference with error handling...")
        left_predictions, left_failed = Inference(test_subset, vis=False, organ='left')
        
        logging.info(f"Left lung results: {len(left_predictions)} successful, {len(left_failed)} failed")
        if left_failed:
            logging.info(f"Failed samples: {left_failed}")
            
        # Summary
        total_attempted = len(test_subset) * 2  # Both lungs
        total_successful = len(right_predictions) + len(left_predictions)
        total_failed = len(right_failed) + len(left_failed)
        
        logging.info(f"\n=== TEST SUMMARY ===")
        logging.info(f"Total samples attempted: {total_attempted} ({len(test_subset)} samples × 2 lungs)")
        logging.info(f"Total successful: {total_successful}")
        logging.info(f"Total failed: {total_failed}")
        logging.info(f"Success rate: {(total_successful/total_attempted)*100:.1f}%")
        
        if total_failed == 0:
            logging.info("✅ All samples processed successfully - error handling system is ready!")
        else:
            logging.info(f"⚠️  {total_failed} samples failed - error handling system caught them properly!")
            
        return True
        
    except Exception as e:
        logging.error(f"Test failed with error: {str(e)}")
        return False
    
    finally:
        # Cleanup temporary file
        if os.path.exists('test_subset_dataset.json'):
            os.remove('test_subset_dataset.json')

if __name__ == "__main__":
    success = test_small_subset()
    if success:
        logging.info("Error handling verification test completed successfully!")
    else:
        logging.error("Error handling verification test failed!")
        sys.exit(1)