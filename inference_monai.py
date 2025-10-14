import torch
from util import ConfigFile, get_sincos_size_embed, txt2embed
import models.m3fm as m3fm
from visualize import Visualize
from ct_prep import get_dataloader
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import logging
from datetime import datetime
import traceback

def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    return obj



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


def Inference(input_data, vis=False):
    collect_predictions = []
    failed_files = []
    args = ConfigFile(input_data['config_file'])

    # Setup logging
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'inference_errors_{args.lung_side}_{log_timestamp}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting inference for {args.lung_side} lung")

    torch.backends.cudnn.benchmark = True
    # create model
    args.modalities = list(args.modalities.split(","))
    model = m3fm.__dict__[args.model](**vars(args))
    model.cuda(args.gpu)

    state_dict = torch.load(args.model_weight, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)

    ## this is the point were we need to change the code

    try:
        data_loader, input_dict = get_dataloader(input_data, args)
        logging.info(f"Successfully loaded dataloader with {len(data_loader)} samples")
    except Exception as e:
        logging.error(f"Failed to create dataloader: {str(e)}")
        raise e

    successful_count = 0
    failed_count = 0

    # Create a robust iterator that handles individual sample failures
    data_iter = iter(data_loader)
    sample_idx = 0
    total_samples = len(data_loader)
    
    with tqdm(total=total_samples, desc=f"Processing {args.lung_side} lung") as pbar:
        while sample_idx < total_samples:
            try:
                # Try to get the next batch - this is where transform errors occur
                batch = next(data_iter)
                
                batch['gpu'] = args.gpu
                batch['data_name'] = [args.data_name]
                
                # Extract file path for error logging
                image_path = batch.get('image_meta_dict', {}).get('filename_or_obj', ['Unknown'])[0] if 'image_meta_dict' in batch else 'Unknown'
                pid = batch.get('pid', ['Unknown'])[0] if 'pid' in batch else 'Unknown'
                

                
                if vis:
                    output_dict, txt_html, img, img_att = Visualize(model, batch, args)

                    for k, v in output_dict.items():
                        output_dict[k] = list(v.cpu().squeeze().numpy())

                    show_image_att(img, img_att)
                    print(output_dict)
                else:
                    model.eval()
                    with torch.no_grad():
                        output_dict = model(batch)

                    for k, v in output_dict.items():
                        output_dict[k] = list(v.cpu().squeeze().numpy())

                    output_dict.update({
                        'pid': batch['pid'][0],
                        'study': batch['study'][0],
                        'series': batch['series'][0],
                        'screen_timepoint': int(batch['screen_timepoint'][0]),
                        'cancer_laterality': [
                            int(batch['cancer_laterality'][0][0]),
                            bool(batch['cancer_laterality'][1][0])
                        ],
                        'y': int(batch['y'][0]),
                        'time_at_event': int(batch['time_at_event'][0]),
                        'y_seq': [int(y) for y in batch['y_seq'][0]],
                        'y_mask': [int(y) for y in batch['y_mask'][0]],
                        'censors': ([int(c) for c in batch['censors'][0]] 
                                  if 'censors' in batch 
                                  else []),
                        'golds': ([int(g) for g in batch['golds'][0]] 
                                if 'golds' in batch 
                                else [])
                    })

                    collect_predictions.append(output_dict)
                
                successful_count += 1
                sample_idx += 1
                pbar.update(1)
                
            except StopIteration:
                # Normal end of data iteration
                break
                
            except Exception as e:
                failed_count += 1
                sample_idx += 1
                pbar.update(1)
                
                # Try to extract information even from failed batch
                try:
                    image_path = batch.get('image_meta_dict', {}).get('filename_or_obj', ['Unknown'])[0] if 'batch' in locals() and 'image_meta_dict' in batch else 'Unknown'
                    pid = batch.get('pid', ['Unknown'])[0] if 'batch' in locals() and 'pid' in batch else 'Unknown'
                except:
                    image_path = 'Unknown (transform failed before batch creation)'
                    pid = 'Unknown'
                
                error_info = {
                    'batch_index': sample_idx - 1,
                    'image_path': image_path,
                    'pid': pid,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                failed_files.append(error_info)
                
                logging.error(f"Failed processing sample {sample_idx - 1} (PID: {pid}, Path: {image_path}): {str(e)}")
                logging.debug(f"Full traceback: {traceback.format_exc()}")
                
                # Continue with next sample instead of failing
                continue
    logging.info(f"Inference completed: {successful_count} successful, {failed_count} failed")
    
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        return obj

    # Convert predictions to JSON serializable format
    collect_predictions = [
        {k: convert_to_json_serializable(v) for k, v in pred.items()}
        for pred in collect_predictions
    ]

    # Save predictions as json
    output_file = f'output_{args.lung_side}.json'
    with open(output_file, 'w') as f:
        json.dump(collect_predictions, f, indent=4)
    logging.info(f"Saved {len(collect_predictions)} predictions to {output_file}")
    
    # Save failed files information
    if failed_files:
        failed_file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        failed_txt_file = f'failed_files_{args.lung_side}_{failed_file_timestamp}.txt'
        failed_json_file = f'failed_files_{args.lung_side}_{failed_file_timestamp}.json'
        
        # Save simple list of failed image paths to txt file
        with open(failed_txt_file, 'w') as f:
            f.write(f"Failed files during {args.lung_side} lung inference - {datetime.now()}\n")
            f.write(f"Total failed: {failed_count}\n\n")
            for fail_info in failed_files:
                f.write(f"PID: {fail_info['pid']}, Path: {fail_info['image_path']}\n")
        
        # Save detailed error information to json file
        with open(failed_json_file, 'w') as f:
            json.dump(failed_files, f, indent=4)
            
        logging.info(f"Saved {len(failed_files)} failed file records to {failed_txt_file} and {failed_json_file}")
    
    return collect_predictions, failed_files

            #return output_dict
