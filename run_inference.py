from inference_monai import Inference
import matplotlib.pyplot as plt
import numpy as np
import json


def show_image_att(imgs, img_atts):
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


## cancer risk demo
input_data = {
    'clinical_txt': "No patient information available.",  # clinical text
    'question': 'Predict the lung cancer risk over six years.',
    'config_file': 'config_files/config_m3mf_cancer_risk.py'
}

Inference(input_data, vis=False)

# Load both JSON files
with open('output_left_npy.json', 'r') as f:
    left_data = json.load(f)

with open('output_right_npy.json', 'r') as f:
    right_data = json.load(f)

# Create dictionaries for lookup by series ID
left_dict = {entry['series']: entry for entry in left_data}
right_dict = {entry['series']: entry for entry in right_data}

# Find common series IDs (intersection)
left_series = set(left_dict.keys())
right_series = set(right_dict.keys())
common_series = left_series & right_series

# Create combined results list
combined_results = []

# Process only entries that exist in both left and right
for series_id in common_series:
    left_entry = left_dict[series_id]
    right_entry = right_dict[series_id]
    
    # Take element-wise maximum of cancer risks
    combined_risks = [max(l, r) for l, r in zip(left_entry['cancer_risk'], right_entry['cancer_risk'])]
    
    # Create combined entry
    combined_entry = {
        'cancer_risk': combined_risks,
        'pid': left_entry['pid'],
        'study': left_entry['study'],
        'series': left_entry['series'],
        # 'exam': left_entry['exam'],
        # 'accession': left_entry['accession'],
        'screen_timepoint': left_entry['screen_timepoint'],
        # 'device': left_entry['device'],
        # 'institution': left_entry['institution'],
        'cancer_laterality': left_entry['cancer_laterality'],
        'y': left_entry['y'],
        'time_at_event': left_entry['time_at_event'],
        'y_seq': left_entry['y_seq'],
        'y_mask': left_entry['y_mask']
    }
    
    combined_results.append(combined_entry)

# Save combined results
with open('output_combined.json', 'w') as f:
    json.dump(combined_results, f, indent=4)

# Print summary
left_only = left_series - right_series
right_only = right_series - left_series

print("\n=== Merge Summary ===")
print(f"Total left entries: {len(left_data)}")
print(f"Total right entries: {len(right_data)}")
print(f"Common series (both lungs): {len(common_series)}")
print(f"Left-only series (no right match): {len(left_only)}")
print(f"Right-only series (no left match): {len(right_only)}")
print(f"Combined results saved: {len(combined_results)}")
print(f"Success rate: {len(combined_results)/(len(left_data) + len(right_data) - len(common_series))*100:.1f}%")

if left_only:
    print(f"\nFirst 5 left-only series:")
    for i, series in enumerate(sorted(left_only)[:5]):
        print(f"  {i+1}. {series}")
    if len(left_only) > 5:
        print(f"  ... and {len(left_only) - 5} more")

if right_only:
    print(f"\nFirst 5 right-only series:")
    for i, series in enumerate(sorted(right_only)[:5]):
        print(f"  {i+1}. {series}")
    if len(right_only) > 5:
        print(f"  ... and {len(right_only) - 5} more")



