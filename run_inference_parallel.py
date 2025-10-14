#!/usr/bin/env python3
"""
Lung Cancer Risk Inference Script
Support for parallel execution of left/right lung inference

Usage:
  python run_inference.py left    # Run left lung only
  python run_inference.py right   # Run right lung only  
  python run_inference.py both    # Run both lungs sequentially
  python run_inference.py both --combine  # Run both and combine results
"""

from inference_monai import Inference
import matplotlib.pyplot as plt
import numpy as np
import json
import logging
from datetime import datetime
import argparse
import sys


def show_image_att(imgs, img_atts):
    """Display image attention visualization"""
    for s in range(0, imgs.shape[0], 4):
        img = imgs[s]
        img_att = img_atts[s]
        fig, axs = plt.subplots(1, 2)
        img = (img - img.min()) / (img.max() - img.min())
        img = np.array([img, img, img]).transpose([1, 2, 0])
        axs[0].imshow(img)
        axs[0].axis('off')

        axs[1].imshow(img_att)
        axs[1].axis('off')
        plt.savefig(f'output{s}.png')
    plt.show()


def get_lung_config(lung_side):
    """Get configuration for specified lung side"""
    if lung_side.lower() == 'left':
        return {
            'clinical_txt': "No patient information available.",
            'question': 'Predict the lung cancer risk over six years.',
            'config_file': 'config_files/config_m3mf_cancer_risk_left.py',
        }
    elif lung_side.lower() == 'right':
        return {
            'clinical_txt': "No patient information available.",
            'question': 'Predict the lung cancer risk over six years.',
            'config_file': 'config_files/config_m3mf_cancer_risk.py',
        }
    else:
        raise ValueError(f"Invalid lung_side: {lung_side}. Must be 'left' or 'right'")


def setup_logging(lung_side):
    """Setup logging for specified lung side"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'{lung_side}_lung_inference_{timestamp}.log'
    
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


def run_single_lung_inference(lung_side):
    """Run inference for a single lung side"""
    
    # Get configuration for this lung
    input_data = get_lung_config(lung_side)
    
    # Setup logging with lung-specific filename
    timestamp = setup_logging(lung_side)
    
    logging.info(f"Starting {lung_side.upper()} lung inference pipeline")
    logging.info(f"Config file: {input_data['config_file']}")
    
    try:
        # Run inference
        predictions, failed = Inference(input_data, vis=False)
        
        logging.info(f"{lung_side.title()} lung: {len(predictions)} successful, {len(failed)} failed")
        
        # The Inference function should have created the output file
        output_file = f'output_{lung_side}.json'
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            logging.info(f"Loaded {len(data)} predictions from {output_file}")
        except FileNotFoundError:
            logging.error(f"{lung_side.title()} lung output file not found: {output_file}")
            logging.info(f"Using in-memory results: {len(predictions)} predictions")
            data = predictions
        
        logging.info(f"{lung_side.title()} lung inference completed successfully")
        return data, failed
        
    except Exception as e:
        logging.error(f"Critical error during {lung_side} lung inference: {str(e)}")
        raise e


def combine_results():
    """Combine left and right lung results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logging.info("=== Starting result combination ===")
    
    # Load JSON files
    try:
        with open('output_left.json', 'r') as f:
            left_data = json.load(f)
        logging.info(f"Loaded {len(left_data)} left lung predictions")
    except FileNotFoundError:
        logging.error("Left lung output file not found")
        return None
    
    try:
        with open('output_right.json', 'r') as f:
            right_data = json.load(f)
        logging.info(f"Loaded {len(right_data)} right lung predictions")
    except FileNotFoundError:
        logging.error("Right lung output file not found")
        return None
    
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
    with open('output_combined.json', 'w') as f:
        json.dump(combined_results, f, indent=4)
    
    # Save partial results separately
    if partial_results:
        with open(f'output_partial_{timestamp}.json', 'w') as f:
            json.dump(partial_results, f, indent=4)
        logging.info(f"Saved {len(partial_results)} partial results (single lung only) to output_partial_{timestamp}.json")
    
    logging.info(f"Saved {len(combined_results)} complete results (both lungs) to output_combined.json")
    logging.info("Combined inference pipeline completed successfully")
    
    return combined_results, partial_results


def run_both_lungs_sequential(combine_results_flag=True):
    """Run both lungs sequentially (original behavior)"""
    
    # Setup main logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'combined_inference_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )

    logging.info("Starting combined left/right lung inference pipeline")

    try:
        # Run inference for each lung with appropriate config
        logging.info("=== Starting LEFT lung inference ===")
        left_data, left_failed = run_single_lung_inference('left')
        
        logging.info("=== Starting RIGHT lung inference ===")
        right_data, right_failed = run_single_lung_inference('right')
        
        logging.info(f"Left lung: {len(left_data)} successful, {len(left_failed)} failed")
        logging.info(f"Right lung: {len(right_data)} successful, {len(right_failed)} failed")
        
        # Combine results if requested
        if combine_results_flag:
            combined_results, partial_results = combine_results()
            return combined_results, partial_results
        else:
            return left_data, right_data
            
    except Exception as e:
        logging.error(f"Critical error during inference: {str(e)}")
        raise e


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Run lung cancer risk inference')
    parser.add_argument('lung_side', 
                       choices=['left', 'right', 'both'],
                       help='Which lung(s) to process: left, right, or both')
    parser.add_argument('--combine', 
                       action='store_true',
                       help='Combine left and right results (only valid when lung_side=both)')
    
    args = parser.parse_args()
    
    if args.lung_side == 'both':
        # Run both lungs sequentially
        results = run_both_lungs_sequential(args.combine)
        if args.combine:
            combined_results, partial_results = results
            print(f"✅ Combined inference completed!")
            print(f"📊 Complete results (both lungs): {len(combined_results)}")
            print(f"📊 Partial results (single lung): {len(partial_results)}")
            print(f"📁 Combined output: output_combined.json")
        else:
            left_data, right_data = results
            print(f"✅ Both lung inference completed!")
            print(f"📊 Left lung results: {len(left_data)}")
            print(f"📊 Right lung results: {len(right_data)}")
            print(f"📁 Outputs: output_left.json, output_right.json")
    else:
        # Run single lung (for parallel execution)
        data, failed = run_single_lung_inference(args.lung_side)
        print(f"✅ {args.lung_side.title()} lung inference completed!")
        print(f"📊 Successful predictions: {len(data)}")
        print(f"📊 Failed samples: {len(failed)}")
        print(f"📁 Output saved to: output_{args.lung_side}.json")


if __name__ == "__main__":
    main()