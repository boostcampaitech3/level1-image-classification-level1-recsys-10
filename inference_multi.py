import argparse
import os
from importlib import import_module
from re import A

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
from sklearn.ensemble import VotingClassifier
from scipy.stats import mode

def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(k, data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = {}
    for task in ['gender', 'mask', 'age']:
        if task == 'gender':
            num_classes = 2
        else:
            num_classes = 3
        
        temp_path = os.path.join(model_dir, 'k'+str(k), 'effnet_' + task)
        model[task] = load_model(temp_path, num_classes, device).to(device).eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    temp_pred = {}
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            for task in ['gender', 'mask', 'age']:
                temp_pred[task] = torch.argmax(model[task](images), dim=-1)

            pred = MaskBaseDataset.encode_multi_class(temp_pred['mask'], temp_pred['gender'], temp_pred['age'])
            preds.extend(pred.cpu().numpy())

    return info, preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for validing (default: 256)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size for image when you trained (default: (224, 224))')
    parser.add_argument('--model', type=str, default='Effnet', help='model type (default: BaseModel)')
    

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', 'code/_main/model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', 'code/_main/output'))
    parser.add_argument('--kfold', type=int, default=5, help='set kfold num (default:5)')

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    fold_output = []
    for k in range(args.kfold) :
        info, preds = inference(k, data_dir, model_dir, output_dir, args)
        print(preds)
        fold_output.append(preds)
        print(f'k{k} Inference done')
    fold_output = np.array(fold_output)
    result = mode(fold_output, axis=0)[0]
    info['ans'] = result[0]
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
