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
with open('output_left.json', 'r') as f:
    left_data = json.load(f)

with open('output_right.json', 'r') as f:
    right_data = json.load(f)

# Ensure both lists have same length
assert len(left_data) == len(right_data), "Left and right data must have same number of entries"

# Create combined results list
combined_results = []

# Process corresponding entries
for left_entry, right_entry in zip(left_data, right_data):
    # Verify we're combining corresponding entries
    assert left_entry['series'] == right_entry['series'], "Series IDs don't match"
    
    # Take element-wise maximum of cancer risks
    combined_risks = [max(l, r) for l, r in zip(left_entry['cancer_risk'], right_entry['cancer_risk'])]
    
    # Create combined entry
    combined_entry = {
        'cancer_risk': combined_risks,
        'pid': left_entry['pid'],
        'study': left_entry['study'],
        'series': left_entry['series']
    }
    
    combined_results.append(combined_entry)

# Save combined results
with open('output_combined.json', 'w') as f:
    json.dump(combined_results, f, indent=4)



