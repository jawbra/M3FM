#!/usr/bin/env python3
"""
Merge left and right lung inference outputs based on series ID matching
"""

import json
import logging
from datetime import datetime

def merge_lung_outputs(left_file, right_file, output_file, partial_file=None):
    """
    Merge left and right lung inference outputs based on series ID
    
    Args:
        left_file: Path to left lung output JSON
        right_file: Path to right lung output JSON  
        output_file: Path to save combined output JSON
        partial_file: Path to save partial results (optional)
    """
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("Starting merge of left and right lung outputs")
    
    # Load the JSON files
    try:
        with open(left_file, 'r') as f:
            left_data = json.load(f)
        logging.info(f"Loaded {len(left_data)} left lung predictions")
    except FileNotFoundError:
        logging.error(f"Left lung output file not found: {left_file}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in left lung file: {left_file}")
        return None
    
    try:
        with open(right_file, 'r') as f:
            right_data = json.load(f)
        logging.info(f"Loaded {len(right_data)} right lung predictions")
    except FileNotFoundError:
        logging.error(f"Right lung output file not found: {right_file}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in right lung file: {right_file}")
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
        
        # Add right lung specific data as additional fields
        combined_entry['left_cancer_risk'] = left_entry['cancer_risk']
        combined_entry['right_cancer_risk'] = right_entry['cancer_risk']
        
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
        logging.error(f"Failed to save combined results: {str(e)}")
        return None
    
    # Save partial results if there are any
    if partial_results:
        if partial_file is None:
            partial_file = f'output_partial_{timestamp}.json'
        
        try:
            with open(partial_file, 'w') as f:
                json.dump(partial_results, f, indent=4)
            logging.info(f"Saved {len(partial_results)} partial results (single lung only) to {partial_file}")
        except Exception as e:
            logging.error(f"Failed to save partial results: {str(e)}")
    
    # Summary statistics
    total_samples = len(combined_results) + len(partial_results)
    success_rate = (len(combined_results) / total_samples * 100) if total_samples > 0 else 0
    
    logging.info("=== MERGE SUMMARY ===")
    logging.info(f"Total unique samples: {total_samples}")
    logging.info(f"Complete (both lungs): {len(combined_results)} ({len(combined_results)/total_samples*100:.1f}%)")
    logging.info(f"Partial (one lung): {len(partial_results)} ({len(partial_results)/total_samples*100:.1f}%)")
    logging.info(f"  - Left only: {len(left_only)}")
    logging.info(f"  - Right only: {len(right_only)}")
    
    return {
        'combined_results': combined_results,
        'partial_results': partial_results,
        'stats': {
            'total_samples': total_samples,
            'complete_samples': len(combined_results),
            'partial_samples': len(partial_results),
            'left_only': len(left_only),
            'right_only': len(right_only),
            'success_rate': success_rate
        }
    }

if __name__ == "__main__":
    # Default file paths - using safe names to avoid overwriting
    left_file = "/home/brandtj/Documents/projects/iderha/M3FM/output_left.json"
    right_file = "/home/brandtj/Documents/projects/iderha/M3FM/output_right.json"
    
    # Safe output filenames with timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/home/brandtj/Documents/projects/iderha/M3FM/merged_combined_{timestamp}.json"
    partial_file = f"/home/brandtj/Documents/projects/iderha/M3FM/merged_partial_{timestamp}.json"
    
    # Perform the merge
    result = merge_lung_outputs(left_file, right_file, output_file, partial_file)
    
    if result:
        print("✅ Merge completed successfully!")
        print(f"📊 Combined {result['stats']['complete_samples']} samples with both lungs")
        print(f"📊 Partial results: {result['stats']['partial_samples']} samples")
        print(f"📁 Output saved to: {output_file}")
    else:
        print("❌ Merge failed!")