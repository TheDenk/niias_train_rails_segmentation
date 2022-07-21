import os
from argparse import ArgumentParser

import gdown
from tqdm import tqdm

models_storage = [
    {
        'name': 'unetplus_resnet34_0.pt',
        'id': '1S-UeCavyHUtX7GBT1Uijdb2SXYYpRSEh'
    },
    {
        'name': 'unetplus_resnet34_1.pt',
        'id': '1CWk7h0gUDUu4abCiWY7nW16sz4tb_nC1'
    },
    {
        'name': 'unetplus_resnet34_2.pt',
        'id': '17S7t8W2gDTt1IQhck8Jqxh-Muzy3lDAb'
    },
    
    {
        'name': 'unet_resnet34_0.pt',
        'id': '11FnssFXa2f6zZEV9COcMJ61rSsjMIITQ'
    },
    {
        'name': 'unet_resnet34_1.pt',
        'id': '1AGOyCzjFxivAxJAkxFY3NJVCYNm19RAe'
    },
    {
        'name': 'unet_resnet34_2.pt',
        'id': '1Aa6z6zG1SjPwhZWZ-od27yvO4a-uWOml'
    },
    
    {
        'name': 'unet_resnext50_32x4d_0.pt',
        'id': '1tAmGGeXlOU7LR5umo3-L8P8ws7qQYSpE'
    },
    {
        'name': 'unet_resnext50_32x4d_1.pt',
        'id': '1_xY0ZgXaR4dELnsAiRDbXZJAkeZffdNO'
    },
    {
        'name': 'unet_resnext50_32x4d_2.pt',
        'id': '1fhW2dx8mCrBF1TL-PCqY5pqTleacnEB8'
    },
    {
        'name': 'unet_resnext50_32x4d_hard_augs_0.pt',
        'id': '1f8EjCPv_OaBgI84pgu1hJkgGkXqjqGGg'
    },
    {
        'name': 'unet_resnext50_32x4d_hard_augs_1.pt',
        'id': '1p0qCZ6tOc_iwF_7upkrYniRBDGiEFFVf'
    },
    {
        'name': 'unet_resnext50_32x4d_hard_augs_2.pt',
        'id': '1LTfxvOmDVb5b174c0ITz0GpbGI7xZAfY'
    },
]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--out_dir', default='models')
    args = parser.parse_args()
    
    url_template = 'https://drive.google.com/uc?id={}'
    os.makedirs(args.out_dir, exist_ok=True)

    for item in tqdm(models_storage):
        out_name = os.path.join(args.out_dir, item['name'])
        url = url_template.format(item['id'])
        gdown.download(url, out_name, quiet=True)