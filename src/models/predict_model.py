import os
import sys

import numpy as np
import torch
import argparse
from PIL.Image import Resampling
from PIL import Image
from torchvision.transforms import ToTensor


sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))


def super_resolve(input_filename: str, output_filename: str, model_path: str):
    """
    функция для увеличения разрешения изображения
    :param input_filename: имя файла
    :param output_filename: имя для upscale-изображения
    :param model_path: путь до обученной модели
    :return:
    """
    model = torch.load(model_path)

    img = Image.open(input_filename).convert('YCbCr')
    y, cb, cr = img.split()

    img_to_tensor = ToTensor()
    input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

    out = model(input)
    out = out.cpu()
    out_img_y = out[0].detach().numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    # модель обучена только для яркости, поэтому остальные компоненты
    # проходят через интерполяцию.
    # реализация для RGB представлена в ноутбуке
    out_img_cb = cb.resize(out_img_y.size, Resampling.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Resampling.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

    out_img.save(output_filename)
    print('output image saved to ', output_filename)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input', dest='input', type=str, required=True)
    arg_parser.add_argument('--output', dest='output', type=str, required=True)
    arg_parser.add_argument('--model', dest='model', type=str, required=True)
    args = arg_parser.parse_args()

    super_resolve(args.input, args.output, args.model)
