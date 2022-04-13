import tarfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from PIL import Image
import torch.utils.data as data
from os import listdir, makedirs, remove
from os.path import join, exists, basename

from PIL.Image import Resampling
from six.moves import urllib
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor


def is_image_file(filename: str) -> bool:
    """
    проверка на то, что файл с расширением изображения
    :param filename: имя файла
    :return: bool
    """
    return any(filename.endswith(extention) for extention in ['.png', '.jpg', '.jpeg'])


def load_image(filepath: str) -> np.ndarray:
    """
    загрузка изображения и разложение по каналам (яркость/синий/красный).
    в модель идет 1 канал (яркость), но можно переписать код под RGB
    :param filepath: имя файла
    :return: первый канал по YCbCr (яркость)
    """
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir: str, input_transform=None, target_transform=None):
        """
        создает список из файлов в директории
        :param image_dir: директория с изображениями
        :param input_transform: функция преобразования входа (Compose for e.x.)
        :param target_transform: функция преобразования таргета (Compose for e.x.)
        """
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """
        возвращает размер выборки
        :return:
        """
        return len(self.image_filenames)

    def __getitem__(self, item: int) -> tuple:
        """
        итератор для прохода по выборке. применяет преобразования ко входу и таргету
        :param item: индекс
        :return: пара из входа и таргета (тензоры)
        """
        input = load_image(self.image_filenames[item])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        return input, target


def download_bsd300(dest: str = 'dataset') -> str:
    """
    загружает датасет, если его еще нет
    :param dest: имя папки для изображений
    :return: полный путь до изображений
    """
    output_image_dir = join(dest, 'BSDS300/images')

    if not exists(output_image_dir):
        makedirs(output_image_dir)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)
        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


def input_transform(crop_size: int, upscale_factor: int) -> Compose:
    """
    преобразование над входным изображением
    :param crop_size: размер для обрезки
    :param upscale_factor: во сколько раз сжать
    :return: функция с переводом в тензор
    """
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size: int) -> Compose:
    """
    преобразование над таргетом
    :param crop_size: размер для обрезки
    :return: функция с переводом в тензор
    """
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def calculate_valid_crop_size(crop_size: int, upscale_factor: int) -> int:
    """
    вычисление корректного размера обрезки
    :param crop_size: размер обрезки изображения
    :param upscale_factor: во сколько раз будет сжато изображение
    :return: обновленный crop_size
    """
    return crop_size - (crop_size % upscale_factor)


def get_training_set(upscale_factor: int) -> DatasetFromFolder:
    """
    производит загрузку данных и инициализирует преобразования
    :param upscale_factor: параметр сжатия
    :return: класс типа torch.utils.data.Dataset для обучения
    """
    root_dir = download_bsd300()
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


def get_test_set(upscale_factor: int) -> DatasetFromFolder:
    """
    производит загрузку данных и инициализирует преобразования
    :param upscale_factor: параметр сжатия
    :return: класс типа torch.utils.data.Dataset для теста
    """
    root_dir = download_bsd300()
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


def train(epoch: int,
          model: nn.Module,
          training_data_loader: DataLoader,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer):
    """
    функция для обучения на одной эпохе
    :param epoch: номер эпохи
    :param model: модель для обучения
    :param training_data_loader: тренировочный DataLoader
    :param criterion: функция потерь
    :param optimizer: оптимизатор
    :return:
    """
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test(model: nn.Module,
         testing_data_loader: DataLoader,
         criterion: nn.Module):
    """
    функция для теста модели на отложенной выборке
    :param model: модель для теста
    :param testing_data_loader: тестовый DataLoader
    :param criterion: функция потерь
    :return:
    """
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * np.log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch: int, model: nn.Module, history_dir: str):
    """
    функция для сохранения состояния модели после эпохи
    :param epoch: номер эпохи
    :param model: модель
    :param history_dir: имя директории для образов
    :return:
    """
    if not exists(history_dir):
        makedirs(history_dir)
    model_out_path = join(history_dir, "model_epoch_{}.pth".format(epoch))
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


class SRCNN(nn.Module):
    def __init__(self, upscale_factor: int):
        """
        модель для увеличения разрешения. реализация может быть переписана для 3-х каналов
        :param upscale_factor: во сколько раз увеличить
        """
        super(SRCNN, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)  # (*, C * r*r, H, W) -> (*, C, H*r, W*r)

        self._initialize_weights()

    def forward(self, x: torch.Tensor):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


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


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_set = get_training_set(upscale_factor=2)
    test_set = get_test_set(upscale_factor=2)

    batch_size = 32

    training_data_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=batch_size,
                                      shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=2, batch_size=batch_size,
                                     shuffle=False)

    model = SRCNN(upscale_factor=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 5):
        train(epoch, model, training_data_loader, criterion, optimizer)
        test(model, testing_data_loader, criterion)
        checkpoint(epoch, model, 'models_history')

    super_resolve('test.jpg', 'out.jpg', model)
