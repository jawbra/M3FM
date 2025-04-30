import torch
from util import ConfigFile, get_sincos_size_embed, txt2embed
import models.m3fm as m3fm
from visualize import Visualize
from ct_prep import get_dataloader
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

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
    args = ConfigFile(input_data['config_file'])

    torch.backends.cudnn.benchmark = True
    # create model
    args.modalities = list(args.modalities.split(","))
    model = m3fm.__dict__[args.model](**vars(args))
    model.cuda(args.gpu)

    state_dict = torch.load(args.model_weight, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)

    ## this is the point were we need to change the code

    data_loader, input_dict = get_dataloader(input_data, args)
    for batch in tqdm(data_loader):

        batch['gpu'] = args.gpu
        batch['data_name'] = [args.data_name]

        if vis:
            output_dict, txt_html, img, img_att = Visualize(model, batch, args)

            for k, v in output_dict.items():
                output_dict[k] = list(v.cpu().squeeze().numpy())

            #return output_dict, txt_html, img, img_att
            show_image_att(img, img_att)
            print(output_dict)
        else:
            model.eval()
            with torch.no_grad():
                output_dict = model(batch)

            for k, v in output_dict.items():
                output_dict[k] = list(v.cpu().squeeze().numpy())

            #print(output_dict)
            output_dict.update({
                'pid': batch['pid'][0],
                'study': batch['study'][0],
                'series': batch['series'][0],
                'exam': batch['exam'][0],
                'accession': batch['accession'][0],
                'screen_timepoint': int(batch['screen_timepoint'][0]),
                'device': int(batch['device'][0]),
                'institution': batch['institution'][0],
                'cancer_laterality': [
                    int(batch['cancer_laterality'][0][0]),
                    bool(batch['cancer_laterality'][1][0])
                ],
                'y': int(batch['y'][0]),
                'time_at_event': int(batch['time_at_event'][0]),
                'y_seq': [int(y) for y in batch['y_seq'][0]],
                'y_mask': [int(y) for y in batch['y_mask'][0]]
            })

            collect_predictions.append(output_dict)
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        return obj


    # Replace the JSON dump section with:
    collect_predictions = [
        {k: convert_to_json_serializable(v) for k, v in pred.items()}
        for pred in collect_predictions
    ]

    #save as json
    with open(f'output_{args.lung_side}.json', 'w') as f:
        json.dump(collect_predictions, f, indent=4)

            #return output_dict
