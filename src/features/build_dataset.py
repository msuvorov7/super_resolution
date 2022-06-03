import os
import torch.utils.data as data
from PIL import Image
from src.data.data_load import download_bsd300
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor


def is_image_file(filename: str) -> bool:
    """
    проверка на то, что файл с расширением изображения
    :param filename: имя файла
    :return: bool
    """
    return any(filename.endswith(extention) for extention in ['.png', '.jpg', '.jpeg'])


def load_image(filepath: str) -> Image.Image:
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
        self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_image_file(x)]

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
    train_dir = os.path.join(root_dir, "train")
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
    test_dir = os.path.join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))
