import json
import numpy as np

# Load both JSON files
with open('output_left.json', 'r') as f:
    left_data = json.load(f)

with open('output_right.json', 'r') as f:
    right_data = json.load(f)

# Create a dictionary to store combined results by study ID
combined_results = {}

# Process both left and right data
for data in [left_data, right_data]:
    for entry in data:
        study_id = f"{entry['pid']}_{entry['study']}"
        cancer_risks = entry['cancer_risk']
        
        if study_id not in combined_results:
            combined_results[study_id] = {
                'cancer_risk': cancer_risks,
                'pid': entry['pid'],
                'study': entry['study']
            }
        else:
            # Take element-wise maximum
            combined_risks = [max(a, b) for a, b in zip(combined_results[study_id]['cancer_risk'], cancer_risks)]
            combined_results[study_id]['cancer_risk'] = combined_risks

# Convert back to list format
final_results = list(combined_results.values())

# Save combined results
with open('output_combined.json', 'w') as f:
    json.dump(final_results, f, indent=4)