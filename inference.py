import os
import glob

import cv2
import gdown
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser

import ttach as tta
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset


parser = ArgumentParser()
parser.add_argument('--images_dir')
parser.add_argument('--out_dir')
args = parser.parse_args()


def get_img_names(folder, img_format='png'):
    img_paths = glob.glob(os.path.join(folder, f'*.{img_format}'))
    img_names = [os.path.basename(x) for x in img_paths]
    return img_names

def preprocess_image(image):
    input_img = image.copy()
    img = cv2.resize(input_img, (GLOBAL_CONFIG['IMG_W'], GLOBAL_CONFIG['IMG_H']))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img

def process_multimask2np(image):
    img = image.cpu().clone()
    img = img.permute(1, 2, 0).numpy().astype(bool)
    h, w, c = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for c_index in range(c):
        mask[img[:, :, c_index]] = LABELS[c_index]
    
    return mask

class TestDataset(Dataset):
    def __init__(self, images_names, images_folder, augmentations=None):
        self.images_folder = images_folder
        self.images_names = images_names
        self.augmentations = augmentations

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, index):
        img_name = self.images_names[index]
        img_path = os.path.join(self.images_folder, img_name)
        
        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        
        if self.augmentations is not None:
            image = self.augmentations(image=image)['image']
        
        image = preprocess_image(image)
        return {'image': image, 'image_name': img_name, 'orig_h': h, 'orig_w': w}
    
    
TEST_IMAGES_FOLDER = args.images_dir
TEST_IMG_NAMES = get_img_names(TEST_IMAGES_FOLDER)

LABELS = [0, 6, 7, 10]

GLOBAL_CONFIG = {
    'device': 'cuda:0',
    
    'IMG_H': 512,
    'IMG_W': 512*2,
    
    'predict_folder': args.out_dir,
}

models = [
    {
        'model': smp.Unet(encoder_name="resnet34", classes=len(LABELS)),
        'ckpt_path': './models/unet_resnet34_0.pt',
        'name': 'unet_resnet34',
        'use_tta': False,
        'weight': 0.08333333333333333,
    },
    {
        'model': smp.Unet(encoder_name="resnet34", classes=len(LABELS)),
        'ckpt_path': './models/unet_resnet34_1.pt',
        'name': 'unet_resnet34',
        'use_tta': False,
        'weight': 0.08333333333333333,
    },
    {
        'model': smp.Unet(encoder_name="resnet34", classes=len(LABELS)),
        'ckpt_path': './models/unet_resnet34_2.pt',
        'name': 'unet_resnet34',
        'use_tta': False,
        'weight': 0.08333333333333333,
    },
    {
        'model': smp.Unet(encoder_name="resnext50_32x4d", classes=len(LABELS)),
        'ckpt_path': './models/unet_resnext50_32x4d_0.pt',
        'name': 'fpn_resnext50_32x4d',
        'use_tta': False,
        'weight': 0.08333333333333333,
    },
    {
        'model': smp.Unet(encoder_name="resnext50_32x4d", classes=len(LABELS)),
        'ckpt_path': './models/unet_resnext50_32x4d_1.pt',
        'name': 'fpn_resnext50_32x4d',
        'use_tta': False,
        'weight': 0.08333333333333333,
    },
    {
        'model': smp.Unet(encoder_name="resnext50_32x4d", classes=len(LABELS)),
        'ckpt_path': './models/unet_resnext50_32x4d_2.pt',
        'name': 'fpn_resnext50_32x4d',
        'use_tta': False,
        'weight': 0.08333333333333333,
    },
    {
        'model': smp.UnetPlusPlus(encoder_name='resnet34', classes=len(LABELS)),
        'ckpt_path': './models/unetplus_resnet34_0.pt',
        'name': 'unetplus_resnet34',
        'use_tta': False,
        'weight': 0.08333333333333333,
    },
    {
        'model': smp.UnetPlusPlus(encoder_name='resnet34', classes=len(LABELS)),
        'ckpt_path': './models/unetplus_resnet34_1.pt',
        'name': 'unetplus_resnet34',
        'use_tta': False,
        'weight': 0.08333333333333333,
    },
    {
        'model': smp.UnetPlusPlus(encoder_name='resnet34', classes=len(LABELS)),
        'ckpt_path': './models/unetplus_resnet34_2.pt',
        'name': 'unetplus_resnet34',
        'use_tta': False,
        'weight': 0.08333333333333333,
    },
    {
        'model': smp.Unet(encoder_name="resnext50_32x4d", classes=len(LABELS)),
        'ckpt_path': './models/unet_resnext50_32x4d_hard_augs_0.pt',
        'name': 'fpn_resnext50_32x4d',
        'use_tta': False,
        'weight': 0.08333333333333333,
    },
    {
        'model': smp.Unet(encoder_name="resnext50_32x4d", classes=len(LABELS)),
        'ckpt_path': './models/unet_resnext50_32x4d_hard_augs_1.pt',
        'name': 'fpn_resnext50_32x4d',
        'use_tta': False,
        'weight': 0.08333333333333333,
    },
    {
        'model': smp.Unet(encoder_name="resnext50_32x4d", classes=len(LABELS)),
        'ckpt_path': './models/unet_resnext50_32x4d_hard_augs_2.pt',
        'name': 'fpn_resnext50_32x4d',
        'use_tta': False,
        'weight': 0.08333333333333333,
    },
]


transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
#         tta.Rotate90(angles=[0, 90, 180]),
        tta.Scale(scales=[1, 2, 4]),
        tta.Multiply(factors=[0.9, 1.1]),
    ]
)

for item in models:
    ckpt = torch.load(item['ckpt_path'])
    item['model'].load_state_dict(ckpt['state_dict'])
    item['model'] = item['model'].eval().to(GLOBAL_CONFIG['device'])
    
    if item['use_tta']:
        item['model'] = tta.SegmentationTTAWrapper(item['model'], transforms, merge_mode='mean')
        
test_dataset = TestDataset(TEST_IMG_NAMES, TEST_IMAGES_FOLDER)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False, drop_last=False)

if __name__ == '__main__':
    os.makedirs(GLOBAL_CONFIG['predict_folder'], exist_ok=True)
    for n, batch in enumerate(tqdm(test_loader)):

        predict = torch.zeros(len(LABELS), GLOBAL_CONFIG['IMG_H'], GLOBAL_CONFIG['IMG_W'], dtype=torch.float32, device='cpu')

        for item in models:
            with torch.no_grad():
                pr = item['model'](batch['image'].to(GLOBAL_CONFIG['device']))
                pr = F.softmax(pr.cpu().detach(), dim=1)[0]
                predict += pr * item['weight']

        predict = process_multimask2np(predict.round())
        predict = Image.fromarray(predict).resize((batch['orig_w'][0], batch['orig_h'][0]), Image.NEAREST)
        img_path = os.path.join(GLOBAL_CONFIG["predict_folder"], batch["image_name"][0])
        predict.save(img_path)