#!/usr/bin/env python3#!/usr/bin/env python3

""""""

Lung Cancer Risk Inference ScriptLung Cancer Risk Inference Script

Updated to support parallel executionUpdated to support parallel execution



Usage Examples:Usage Examples:

  # For parallel execution in separate terminals:  # For parallel execution in separate terminals:

  Terminal 1: python run_inference.py left  Terminal 1: python run_inference.py left

  Terminal 2: python run_inference.py right  Terminal 2: python run_inference.py right

    

  # For sequential execution:  # For sequential execution:

  python run_inference.py both  python run_inference.py both

  python run_inference.py both --combine  python run_inference.py both --combine

""""""



from inference_monai import Inferencefrom inference_monai import Inference

import jsonimport json

import loggingimport logging

from datetime import datetimefrom datetime import datetime

import argparseimport argparse





def get_lung_config(lung_side):def get_lung_config(lung_side):

    """Get configuration for specified lung side"""    """Get configuration for specified lung side"""

    if lung_side.lower() == 'left':    if lung_side.lower() == 'left':

        return {        return {

            'clinical_txt': "No patient information available.",            'clinical_txt': "No patient information available.",

            'question': 'Predict the lung cancer risk over six years.',            'question': 'Predict the lung cancer risk over six years.',

            'config_file': 'config_files/config_m3mf_cancer_risk_left.py',            'config_file': 'config_files/config_m3mf_cancer_risk_left.py',

        }        }

    elif lung_side.lower() == 'right':    elif lung_side.lower() == 'right':

        return {        return {

            'clinical_txt': "No patient information available.",            'clinical_txt': "No patient information available.",

            'question': 'Predict the lung cancer risk over six years.',            'question': 'Predict the lung cancer risk over six years.',

            'config_file': 'config_files/config_m3mf_cancer_risk.py',            'config_file': 'config_files/config_m3mf_cancer_risk.py',

        }        }

    else:    else:

        raise ValueError(f"Invalid lung_side: {lung_side}. Must be 'left' or 'right'")        raise ValueError(f"Invalid lung_side: {lung_side}. Must be 'left' or 'right'")





def setup_logging(lung_side):def setup_logging(lung_side):

    """Setup logging for specified lung side"""    """Setup logging for specified lung side"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_filename = f'{lung_side}_lung_inference_{timestamp}.log'    log_filename = f'{lung_side}_lung_inference_{timestamp}.log'

        

    # Clear any existing handlers    # Clear any existing handlers

    for handler in logging.root.handlers[:]:    for handler in logging.root.handlers[:]:

        logging.root.removeHandler(handler)        logging.root.removeHandler(handler)

        

    logging.basicConfig(    logging.basicConfig(

        level=logging.INFO,        level=logging.INFO,

        format='%(asctime)s - %(levelname)s - %(message)s',        format='%(asctime)s - %(levelname)s - %(message)s',

        handlers=[        handlers=[

            logging.FileHandler(log_filename),            logging.FileHandler(log_filename),

            logging.StreamHandler()            logging.StreamHandler()

        ]        ]

    )    )

        

    return timestamp    return timestamp





def run_single_lung_inference(lung_side):def run_single_lung_inference(lung_side):

    """Run inference for a single lung side"""    """Run inference for a single lung side"""

        

    # Get configuration for this lung    # Get configuration for this lung

    input_data = get_lung_config(lung_side)    input_data = get_lung_config(lung_side)

        

    # Setup logging with lung-specific filename    # Setup logging with lung-specific filename

    timestamp = setup_logging(lung_side)    timestamp = setup_logging(lung_side)

        

    logging.info(f"Starting {lung_side.upper()} lung inference pipeline")    logging.info(f"Starting {lung_side.upper()} lung inference pipeline")

    logging.info(f"Config file: {input_data['config_file']}")    logging.info(f"Config file: {input_data['config_file']}")

        

    try:    try:

        # Run inference        # Run inference

        predictions, failed = Inference(input_data, vis=False)        predictions, failed = Inference(input_data, vis=False)

                

        logging.info(f"{lung_side.title()} lung: {len(predictions)} successful, {len(failed)} failed")        logging.info(f"{lung_side.title()} lung: {len(predictions)} successful, {len(failed)} failed")

        logging.info(f"{lung_side.title()} lung inference completed successfully")        logging.info(f"{lung_side.title()} lung inference completed successfully")

                

        return predictions, failed        return predictions, failed

                

    except Exception as e:    except Exception as e:

        logging.error(f"Critical error during {lung_side} lung inference: {str(e)}")        logging.error(f"Critical error during {lung_side} lung inference: {str(e)}")

        raise e        raise e





def combine_results():def combine_results():

    """Combine left and right lung results"""    """Combine left and right lung results"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        

    logging.info("=== Starting result combination ===")    logging.info("=== Starting result combination ===")

        

    # Load JSON files    # Load JSON files

    try:    try:

        with open('output_left.json', 'r') as f:        with open('output_left.json', 'r') as f:

            left_data = json.load(f)            left_data = json.load(f)

        logging.info(f"Loaded {len(left_data)} left lung predictions")        logging.info(f"Loaded {len(left_data)} left lung predictions")

    except FileNotFoundError:    except FileNotFoundError:

        logging.error("Left lung output file not found")        logging.error("Left lung output file not found")

        return None, None        return None, None

        

    try:    try:

        with open('output_right.json', 'r') as f:        with open('output_right.json', 'r') as f:

            right_data = json.load(f)            right_data = json.load(f)

        logging.info(f"Loaded {len(right_data)} right lung predictions")        logging.info(f"Loaded {len(right_data)} right lung predictions")

    except FileNotFoundError:    except FileNotFoundError:

        logging.error("Right lung output file not found")        logging.error("Right lung output file not found")

        return None, None        return None, None

        

    # Create dictionaries for quick lookup by series ID    # Create dictionaries for quick lookup by series ID

    left_dict = {entry['series']: entry for entry in left_data}    left_dict = {entry['series']: entry for entry in left_data}

    right_dict = {entry['series']: entry for entry in right_data}    right_dict = {entry['series']: entry for entry in right_data}

        

    # Find common series IDs (samples that succeeded in both left and right)    # Find common series IDs (samples that succeeded in both left and right)

    common_series = set(left_dict.keys()) & set(right_dict.keys())    common_series = set(left_dict.keys()) & set(right_dict.keys())

    left_only = set(left_dict.keys()) - set(right_dict.keys())    left_only = set(left_dict.keys()) - set(right_dict.keys())

    right_only = set(right_dict.keys()) - set(left_dict.keys())    right_only = set(right_dict.keys()) - set(left_dict.keys())

        

    logging.info(f"Common series (both lungs successful): {len(common_series)}")    logging.info(f"Common series (both lungs successful): {len(common_series)}")

    logging.info(f"Left-only series (right lung failed): {len(left_only)}")    logging.info(f"Left-only series (right lung failed): {len(left_only)}")

    logging.info(f"Right-only series (left lung failed): {len(right_only)}")    logging.info(f"Right-only series (left lung failed): {len(right_only)}")

        

    # Create combined results list    # Create combined results list

    combined_results = []    combined_results = []

    partial_results = []    partial_results = []

        

    # Process entries that succeeded in both lungs    # Process entries that succeeded in both lungs

    for series_id in common_series:    for series_id in common_series:

        left_entry = left_dict[series_id]        left_entry = left_dict[series_id]

        right_entry = right_dict[series_id]        right_entry = right_dict[series_id]

                

        # Take element-wise maximum of cancer risks        # Take element-wise maximum of cancer risks

        combined_risks = [max(l, r) for l, r in zip(left_entry['cancer_risk'], right_entry['cancer_risk'])]        combined_risks = [max(l, r) for l, r in zip(left_entry['cancer_risk'], right_entry['cancer_risk'])]

                

        # Create combined entry - use left entry as base and update cancer_risk        # Create combined entry - use left entry as base and update cancer_risk

        combined_entry = left_entry.copy()        combined_entry = left_entry.copy()

        combined_entry['cancer_risk'] = combined_risks        combined_entry['cancer_risk'] = combined_risks

        combined_entry['processing_note'] = 'both_lungs_successful'        combined_entry['processing_note'] = 'both_lungs_successful'

                

        combined_results.append(combined_entry)        combined_results.append(combined_entry)

        

    # Handle partial results (only one lung succeeded)    # Handle partial results (only one lung succeeded)

    for series_id in left_only:    for series_id in left_only:

        partial_entry = left_dict[series_id].copy()        partial_entry = left_dict[series_id].copy()

        partial_entry['processing_note'] = 'left_lung_only_right_failed'        partial_entry['processing_note'] = 'left_lung_only_right_failed'

        partial_results.append(partial_entry)        partial_results.append(partial_entry)

        

    for series_id in right_only:    for series_id in right_only:

        partial_entry = right_dict[series_id].copy()        partial_entry = right_dict[series_id].copy()

        partial_entry['processing_note'] = 'right_lung_only_left_failed'        partial_entry['processing_note'] = 'right_lung_only_left_failed'

        partial_results.append(partial_entry)        partial_results.append(partial_entry)

        

    # Save combined results    # Save combined results

    with open('output_combined.json', 'w') as f:    with open('output_combined.json', 'w') as f:

        json.dump(combined_results, f, indent=4)        json.dump(combined_results, f, indent=4)

        

    # Save partial results separately    # Save partial results separately

    if partial_results:    if partial_results:

        with open(f'output_partial_{timestamp}.json', 'w') as f:        with open(f'output_partial_{timestamp}.json', 'w') as f:

            json.dump(partial_results, f, indent=4)            json.dump(partial_results, f, indent=4)

        logging.info(f"Saved {len(partial_results)} partial results (single lung only) to output_partial_{timestamp}.json")        logging.info(f"Saved {len(partial_results)} partial results (single lung only) to output_partial_{timestamp}.json")

        

    logging.info(f"Saved {len(combined_results)} complete results (both lungs) to output_combined.json")    logging.info(f"Saved {len(combined_results)} complete results (both lungs) to output_combined.json")

    logging.info("Combined inference pipeline completed successfully")    logging.info("Combined inference pipeline completed successfully")

        

    return combined_results, partial_results    return combined_results, partial_results





def main():def main():

    """Main function with command line argument parsing"""    """Main function with command line argument parsing"""

    parser = argparse.ArgumentParser(description='Run lung cancer risk inference')    parser = argparse.ArgumentParser(description='Run lung cancer risk inference')

    parser.add_argument('lung_side',     parser.add_argument('lung_side', 

                       choices=['left', 'right', 'both'],                       choices=['left', 'right', 'both'],

                       help='Which lung(s) to process: left, right, or both')                       help='Which lung(s) to process: left, right, or both')

    parser.add_argument('--combine',     parser.add_argument('--combine', 

                       action='store_true',                       action='store_true',

                       help='Combine left and right results (only valid when lung_side=both)')                       help='Combine left and right results (only valid when lung_side=both)')

        

    args = parser.parse_args()    args = parser.parse_args()

        

    print(f"🚀 Starting {args.lung_side} lung inference...")    print(f"🚀 Starting {args.lung_side} lung inference...")

        

    if args.lung_side == 'both':    if args.lung_side == 'both':

        # Setup combined logging        # Setup combined logging

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for handler in logging.root.handlers[:]:        for handler in logging.root.handlers[:]:

            logging.root.removeHandler(handler)            logging.root.removeHandler(handler)

                        

        logging.basicConfig(        logging.basicConfig(

            level=logging.INFO,            level=logging.INFO,

            format='%(asctime)s - %(levelname)s - %(message)s',            format='%(asctime)s - %(levelname)s - %(message)s',

            handlers=[            handlers=[

                logging.FileHandler(f'combined_inference_{timestamp}.log'),                logging.FileHandler(f'combined_inference_{timestamp}.log'),

                logging.StreamHandler()                logging.StreamHandler()

            ]            ]

        )        )

                

        logging.info("Starting combined left/right lung inference pipeline")        logging.info("Starting combined left/right lung inference pipeline")

                

        # Run both lungs sequentially        # Run both lungs sequentially

        logging.info("=== Starting LEFT lung inference ===")        logging.info("=== Starting LEFT lung inference ===")

        left_predictions, left_failed = run_single_lung_inference('left')        left_predictions, left_failed = run_single_lung_inference('left')

                

        logging.info("=== Starting RIGHT lung inference ===")        logging.info("=== Starting RIGHT lung inference ===")

        right_predictions, right_failed = run_single_lung_inference('right')        right_predictions, right_failed = run_single_lung_inference('right')

                

        print(f"📊 Left lung: {len(left_predictions)} successful, {len(left_failed)} failed")        print(f"📊 Left lung: {len(left_predictions)} successful, {len(left_failed)} failed")

        print(f"📊 Right lung: {len(right_predictions)} successful, {len(right_failed)} failed")        print(f"📊 Right lung: {len(right_predictions)} successful, {len(right_failed)} failed")

                

        # Combine results if requested        # Combine results if requested

        if args.combine:        if args.combine:

            combined_results, partial_results = combine_results()            combined_results, partial_results = combine_results()

            if combined_results is not None:            if combined_results is not None:

                print(f"✅ Combined inference completed!")                print(f"✅ Combined inference completed!")

                print(f"📊 Complete results (both lungs): {len(combined_results)}")                print(f"📊 Complete results (both lungs): {len(combined_results)}")

                print(f"📊 Partial results (single lung): {len(partial_results) if partial_results else 0}")                print(f"📊 Partial results (single lung): {len(partial_results) if partial_results else 0}")

                print(f"📁 Combined output: output_combined.json")                print(f"📁 Combined output: output_combined.json")

        else:        else:

            print(f"✅ Both lungs inference completed!")            print(f"✅ Both lungs inference completed!")

            print(f"📁 Outputs: output_left.json, output_right.json")            print(f"📁 Outputs: output_left.json, output_right.json")

            print("💡 Use --combine flag to merge results")            print("💡 Use --combine flag to merge results")

                        

    else:    else:

        # Run single lung (for parallel execution)        # Run single lung (for parallel execution)

        predictions, failed = run_single_lung_inference(args.lung_side)        predictions, failed = run_single_lung_inference(args.lung_side)

        print(f"✅ {args.lung_side.title()} lung inference completed!")        print(f"✅ {args.lung_side.title()} lung inference completed!")

        print(f"📊 Successful predictions: {len(predictions)}")        print(f"📊 Successful predictions: {len(predictions)}")

        print(f"📊 Failed samples: {len(failed)}")        print(f"📊 Failed samples: {len(failed)}")

        print(f"📁 Output saved to: output_{args.lung_side}.json")        print(f"📁 Output saved to: output_{args.lung_side}.json")





if __name__ == "__main__":if __name__ == "__main__":

    main()    main()