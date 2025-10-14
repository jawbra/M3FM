#!/usr/bin/env python3
"""
Combine Left and Right Lung Inference Results
Standalone script to merge output_left.json and output_right.json files

Usage:
    python combine_results.py
    python combine_results.py --left /path/to/output_left.json --right /path/to/output_right.json
"""

import json
import logging
from datetime import datetime
import argparse
import os


def setup_logging():
    """Setup logging for combination process"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'combine_results_{timestamp}.log'
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return timestamp


def combine_results(left_file, right_file, output_file='output_combined.json'):
    """Combine left and right lung results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logging.info("=== Starting result combination ===")
    logging.info(f"Left file: {left_file}")
    logging.info(f"Right file: {right_file}")
    logging.info(f"Output file: {output_file}")
    
    # Check if files exist
    if not os.path.exists(left_file):
        logging.error(f"Left lung output file not found: {left_file}")
        return None, None
    
    if not os.path.exists(right_file):
        logging.error(f"Right lung output file not found: {right_file}")
        return None, None
    
    # Load JSON files
    try:
        with open(left_file, 'r') as f:
            left_data = json.load(f)
        logging.info(f"Loaded {len(left_data)} left lung predictions")
    except Exception as e:
        logging.error(f"Error loading left file: {str(e)}")
        return None, None
    
    try:
        with open(right_file, 'r') as f:
            right_data = json.load(f)
        logging.info(f"Loaded {len(right_data)} right lung predictions")
    except Exception as e:
        logging.error(f"Error loading right file: {str(e)}")
        return None, None
    
    # Create dictionaries for quick lookup by series ID
    left_dict = {entry['series']: entry for entry in left_data}
    right_dict = {entry['series']: entry for entry in right_data}
    
    # Find common series IDs (samples that succeeded in both left and right)
    common_series = set(left_dict.keys()) & set(right_dict.keys())
    left_only = set(left_dict.keys()) - set(right_dict.keys())
    right_only = set(right_dict.keys()) - set(left_dict.keys())
    
    logging.info(f"Common series (both lungs successful): {len(common_series)}")
    logging.info(f"Left-only series (right lung failed): {len(left_only)}")
    logging.info(f"Right-only series (left lung failed): {len(right_only)}")
    
    # Create combined results list
    combined_results = []
    partial_results = []
    
    # Process entries that succeeded in both lungs
    for series_id in common_series:
        left_entry = left_dict[series_id]
        right_entry = right_dict[series_id]
        
        # Take element-wise maximum of cancer risks
        combined_risks = [max(l, r) for l, r in zip(left_entry['cancer_risk'], right_entry['cancer_risk'])]
        
        # Create combined entry - use left entry as base and update cancer_risk
        combined_entry = left_entry.copy()
        combined_entry['cancer_risk'] = combined_risks
        combined_entry['processing_note'] = 'both_lungs_successful'
        
        combined_results.append(combined_entry)
    
    # Handle partial results (only one lung succeeded)
    for series_id in left_only:
        partial_entry = left_dict[series_id].copy()
        partial_entry['processing_note'] = 'left_lung_only_right_failed'
        partial_results.append(partial_entry)
    
    for series_id in right_only:
        partial_entry = right_dict[series_id].copy()
        partial_entry['processing_note'] = 'right_lung_only_left_failed'
        partial_results.append(partial_entry)
    
    # Save combined results
    try:
        with open(output_file, 'w') as f:
            json.dump(combined_results, f, indent=4)
        logging.info(f"Saved {len(combined_results)} complete results (both lungs) to {output_file}")
    except Exception as e:
        logging.error(f"Error saving combined results: {str(e)}")
        return None, None
    
    # Save partial results separately if any exist
    if partial_results:
        partial_output = f'output_partial_{timestamp}.json'
        try:
            with open(partial_output, 'w') as f:
                json.dump(partial_results, f, indent=4)
            logging.info(f"Saved {len(partial_results)} partial results (single lung only) to {partial_output}")
        except Exception as e:
            logging.error(f"Error saving partial results: {str(e)}")
    
    # Summary statistics
    total_predictions = len(combined_results) + len(partial_results)
    success_rate = (len(combined_results) / total_predictions * 100) if total_predictions > 0 else 0
    
    logging.info("=== Combination Summary ===")
    logging.info(f"Total samples processed: {total_predictions}")
    logging.info(f"Both lungs successful: {len(combined_results)} ({len(combined_results)/total_predictions*100:.1f}%)")
    logging.info(f"Single lung only: {len(partial_results)} ({len(partial_results)/total_predictions*100:.1f}%)")
    logging.info("Combination pipeline completed successfully")
    
    return combined_results, partial_results


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Combine left and right lung inference results')
    parser.add_argument('--left', 
                       default='output_left.json',
                       help='Path to left lung output file (default: output_left.json)')
    parser.add_argument('--right', 
                       default='output_right.json',
                       help='Path to right lung output file (default: output_right.json)')
    parser.add_argument('--output', 
                       default='output_combined.json',
                       help='Path for combined output file (default: output_combined.json)')
    
    args = parser.parse_args()
    
    # Setup logging
    timestamp = setup_logging()
    
    print(f"🔄 Combining lung inference results...")
    print(f"📂 Left file: {args.left}")
    print(f"📂 Right file: {args.right}")
    print(f"📁 Output file: {args.output}")
    
    # Run combination
    combined_results, partial_results = combine_results(args.left, args.right, args.output)
    
    if combined_results is not None:
        print(f"✅ Combination completed successfully!")
        print(f"📊 Complete results (both lungs): {len(combined_results)}")
        print(f"📊 Partial results (single lung): {len(partial_results) if partial_results else 0}")
        print(f"📁 Combined output saved to: {args.output}")
        if partial_results:
            print(f"📁 Partial results saved to: output_partial_{timestamp}.json")
    else:
        print(f"❌ Combination failed! Check the log file for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())