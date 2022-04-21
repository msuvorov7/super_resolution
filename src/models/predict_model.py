import numpy as np
import torch.nn as nn
from PIL.Image import Resampling
from PIL import Image
from torchvision.transforms import ToTensor


def super_resolve(input_filename: str, output_filename: str, model: nn.Module):
    """
    функция для увеличения разрешения изображения
    :param input_filename: имя файла
    :param output_filename: имя для upscale-изображения
    :param model: обученная модель
    :return:
    """
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
