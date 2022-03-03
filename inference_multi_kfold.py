import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import MaskBaseDataset, InferenceDataset


def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def load_conv(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), 'MultiClassResnet18')
    model = model_cls(
        num_classes=num_classes
    )

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # num_classes = MaskBaseDataset.num_classes  # 18
    tasks = ['age', 'gender', 'mask']
    folds = ['multitest', 'multitest1', 'multitest2', 'multitest3', 'multitest4']

    ensemble_path = os.path.join(data_dir, 'info.csv')
    img_root = os.path.join(data_dir, 'images')
    ensemble_csv = pd.read_csv(ensemble_path)

    preds = {'age': [], 'gender': [], 'mask': []}
    # for task in tasks:

    model = {}

    # if task == 'gender':
    #     num_classes = 2
    # else:
        # num_classes = 3
    num_classes = 0

    img_paths = [os.path.join(img_root, img_id) for img_id in ensemble_csv.ImageID]

    dataset = InferenceDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    for fold in folds:
        model_path = os.path.join(model_dir,fold)
        model[fold] = load_model(model_path, num_classes, device).to(device)
        model[fold].eval()

    print("Calculating inference results..")
    
    temp_pred = {}
    temp_sum = {'age': 0., 'gender': 0., 'mask': 0.}
    with torch.no_grad():
        for idx, images in enumerate(loader):
            temp_sum = {'age': 0., 'gender': 0., 'mask': 0.}
            images = images.to(device)
            for fold in folds:
                model[fold].to(device)
                for task in tasks:
                    temp_sum[task] += model[fold](images)[task]
            for task in tasks:
                # print(temp_sum[task])
                preds[task].append(torch.argmax(temp_sum[task], dim=-1))
                # print(f'task: {preds[task]}')
    for task in tasks:
        # print(preds[task].size())
        preds[task] = torch.cat(preds[task])
    

    # print(preds[task])

    # print(preds[task].size())

    pred = MaskBaseDataset.encode_multi_class(preds['mask'], preds['gender'], preds['age'])
    ensemble_csv['ans'] = pred.cpu().numpy()
    ensemble_csv.to_csv(os.path.join(output_dir, f'gan.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for validing (default: 256)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size for image when you trained (default: (224, 224))')
    parser.add_argument('--model', type=str, default='MultiClassResnet18', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
