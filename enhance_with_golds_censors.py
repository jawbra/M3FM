#!/usr/bin/env python3
"""
Add golds and censors fields to merged inference results by matching series IDs 
with the original dataset file
"""

import json
import logging
from datetime import datetime
import sys
import os

def create_golds_and_censors(y_seq, y_mask):
    """
    Create golds and censors from y_seq and y_mask
    
    Args:
        y_seq: Ground truth sequence
        y_mask: Mask indicating valid/observed values
    
    Returns:
        tuple: (golds, censors) where golds=y_seq and censors=1-y_mask
    """
    golds = y_seq.copy() if isinstance(y_seq, list) else list(y_seq)
    # Censors: 1 where event is censored (y_mask=0), 0 where event is observed (y_mask=1)
    censors = [1 - mask for mask in (y_mask if isinstance(y_mask, list) else list(y_mask))]
    return golds, censors

def add_golds_censors_to_results(dataset_file, results_file, output_file):
    """
    Add golds and censors fields to inference results by matching series IDs
    
    Args:
        dataset_file: Path to original dataset JSON file
        results_file: Path to merged inference results JSON file  
        output_file: Path to save enhanced results with golds/censors
    """
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("Starting to add golds and censors to inference results")
    
    # Load dataset file
    try:
        logging.info(f"Loading dataset file: {dataset_file}")
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
        logging.info(f"Loaded {len(dataset)} dataset entries")
    except FileNotFoundError:
        logging.error(f"Dataset file not found: {dataset_file}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in dataset file: {dataset_file}")
        return None
    
    # Load results file
    try:
        logging.info(f"Loading results file: {results_file}")
        with open(results_file, 'r') as f:
            results = json.load(f)
        logging.info(f"Loaded {len(results)} inference results")
    except FileNotFoundError:
        logging.error(f"Results file not found: {results_file}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in results file: {results_file}")
        return None
    
    # Create dataset lookup dictionary by series ID
    logging.info("Creating series ID lookup dictionary from dataset...")
    dataset_dict = {}
    missing_series_count = 0
    
    for entry in dataset:
        if 'series' in entry:
            series_id = entry['series']
            dataset_dict[series_id] = entry
        else:
            missing_series_count += 1
    
    logging.info(f"Created lookup for {len(dataset_dict)} series IDs from dataset")
    if missing_series_count > 0:
        logging.warning(f"Found {missing_series_count} dataset entries without series ID")
    
    # Process results and add golds/censors - only keep matched entries
    enhanced_results = []
    matched_count = 0
    no_series_id_count = 0
    series_not_found_count = 0
    missing_outcome_data_count = 0
    
    # Track unmatched entries for reporting
    unmatched_entries = {
        'no_series_id': [],
        'series_not_found': [],
        'missing_outcome_data': []
    }
    
    logging.info("Matching inference results with dataset entries...")
    
    for result in results:
        if 'series' not in result:
            no_series_id_count += 1
            unmatched_entries['no_series_id'].append({
                'pid': result.get('pid', 'Unknown'),
                'reason': 'Result entry missing series ID'
            })
            continue  # Skip this entry - don't include in final results
            
        series_id = result['series']
        
        if series_id not in dataset_dict:
            series_not_found_count += 1
            unmatched_entries['series_not_found'].append({
                'series': series_id,
                'pid': result.get('pid', 'Unknown'),
                'reason': f'No dataset entry found for series {series_id}'
            })
            continue  # Skip this entry - don't include in final results
        
        # Found match in dataset
        dataset_entry = dataset_dict[series_id]
        
        if 'y_seq' not in dataset_entry or 'y_mask' not in dataset_entry:
            missing_outcome_data_count += 1
            unmatched_entries['missing_outcome_data'].append({
                'series': series_id,
                'pid': result.get('pid', 'Unknown'),
                'reason': f'Dataset entry for series {series_id} missing y_seq or y_mask'
            })
            continue  # Skip this entry - don't include in final results
        
        # Successfully matched - create enhanced result
        enhanced_result = result.copy()
        
        golds, censors = create_golds_and_censors(
            dataset_entry['y_seq'], 
            dataset_entry['y_mask']
        )
        
        enhanced_result['golds'] = golds
        enhanced_result['censors'] = censors
        enhanced_result['match_status'] = 'matched'
        
        # Also add original y_seq and y_mask for reference
        enhanced_result['original_y_seq'] = dataset_entry['y_seq']
        enhanced_result['original_y_mask'] = dataset_entry['y_mask']
        
        enhanced_results.append(enhanced_result)
        matched_count += 1
    
    # Save enhanced results
    try:
        with open(output_file, 'w') as f:
            json.dump(enhanced_results, f, indent=4)
        logging.info(f"Saved {len(enhanced_results)} enhanced results to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save enhanced results: {str(e)}")
        return None
    
    # Summary statistics
    total_unmatched = no_series_id_count + series_not_found_count + missing_outcome_data_count
    match_rate = (matched_count / len(results) * 100) if results else 0
    
    logging.info("=== ENHANCEMENT SUMMARY ===")
    logging.info(f"Total results processed: {len(results)}")
    logging.info(f"Successfully matched and included: {matched_count} ({match_rate:.1f}%)")
    logging.info(f"Unmatched entries (excluded): {total_unmatched} ({100-match_rate:.1f}%)")
    logging.info(f"  - Missing series ID: {no_series_id_count}")
    logging.info(f"  - Series not found in dataset: {series_not_found_count}")
    logging.info(f"  - Missing outcome data (y_seq/y_mask): {missing_outcome_data_count}")
    logging.info(f"Dataset entries available: {len(dataset_dict)}")
    logging.info(f"Final enhanced results count: {len(enhanced_results)}")
    
    # Print detailed unmatched report
    logging.info("\n=== DETAILED UNMATCHED REPORT ===")
    
    if unmatched_entries['no_series_id']:
        logging.info(f"\nEntries missing series ID ({no_series_id_count}):")
        for entry in unmatched_entries['no_series_id'][:5]:  # Show first 5
            logging.info(f"  PID: {entry['pid']} - {entry['reason']}")
        if len(unmatched_entries['no_series_id']) > 5:
            logging.info(f"  ... and {len(unmatched_entries['no_series_id']) - 5} more")
    
    if unmatched_entries['series_not_found']:
        logging.info(f"\nSeries not found in dataset ({series_not_found_count}):")
        for entry in unmatched_entries['series_not_found'][:5]:  # Show first 5
            logging.info(f"  Series: {entry['series'][:50]}... (PID: {entry['pid']})")
        if len(unmatched_entries['series_not_found']) > 5:
            logging.info(f"  ... and {len(unmatched_entries['series_not_found']) - 5} more")
    
    if unmatched_entries['missing_outcome_data']:
        logging.info(f"\nMissing outcome data ({missing_outcome_data_count}):")
        for entry in unmatched_entries['missing_outcome_data'][:5]:  # Show first 5
            logging.info(f"  Series: {entry['series'][:50]}... (PID: {entry['pid']})")
        if len(unmatched_entries['missing_outcome_data']) > 5:
            logging.info(f"  ... and {len(unmatched_entries['missing_outcome_data']) - 5} more")
    
    return {
        'enhanced_results': enhanced_results,
        'unmatched_entries': unmatched_entries,
        'stats': {
            'total_results': len(results),
            'matched_count': matched_count,
            'total_unmatched': total_unmatched,
            'no_series_id_count': no_series_id_count,
            'series_not_found_count': series_not_found_count,
            'missing_outcome_data_count': missing_outcome_data_count,
            'match_rate': match_rate,
            'dataset_size': len(dataset_dict),
            'final_count': len(enhanced_results)
        }
    }

def main():
    """Main function to process command line arguments or use defaults"""
    
    # Default file paths
    dataset_file = "/home/brandtj/Documents/projects/iderha/M3FM/test_oct5_selene_modified.json"
    
    # Use the existing combined results file
    results_file = "/home/brandtj/Documents/projects/iderha/M3FM/predictions_old/output_combined.json"
    
    # Check if the file exists
    if not os.path.exists(results_file):
        logging.error(f"Results file not found: {results_file}")
        print(f"❌ Results file not found: {results_file}")
        return
    
    logging.info(f"Using existing combined results file: {results_file}")
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/home/brandtj/Documents/projects/iderha/M3FM/enhanced_combined_results_{timestamp}.json"
    
    # Process the files
    result = add_golds_censors_to_results(dataset_file, results_file, output_file)
    
    if result:
        stats = result['stats']
        print("✅ Enhancement completed successfully!")
        print(f"\n📊 MATCHING SUMMARY:")
        print(f"   • Total input results: {stats['total_results']}")
        print(f"   • Successfully matched: {stats['matched_count']} ({stats['match_rate']:.1f}%)")
        print(f"   • Excluded (unmatched): {stats['total_unmatched']} ({100-stats['match_rate']:.1f}%)")
        print(f"   • Final enhanced results: {stats['final_count']}")
        
        print(f"\n❌ EXCLUSION BREAKDOWN:")
        print(f"   • Missing series ID: {stats['no_series_id_count']}")
        print(f"   • Series not in dataset: {stats['series_not_found_count']}")
        print(f"   • Missing outcome data: {stats['missing_outcome_data_count']}")
        
        print(f"\n📁 Output saved to: {output_file}")
        
        # Show some sample enhanced data
        if result['enhanced_results']:
            sample = result['enhanced_results'][0]
            if 'golds' in sample and sample['golds']:
                print(f"\n📋 SAMPLE DATA:")
                print(f"   • Sample series: {sample['series'][:50]}...")
                print(f"   • Sample golds: {sample['golds']}")
                print(f"   • Sample censors: {sample['censors']}")
                print(f"   • Original y_seq: {sample['original_y_seq']}")
                print(f"   • Original y_mask: {sample['original_y_mask']}")
    else:
        print("❌ Enhancement failed!")

if __name__ == "__main__":
    main()