import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader


def train(epoch: int,
          model: nn.Module,
          training_data_loader: DataLoader,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          device: str):
    """
    функция для обучения на одной эпохе
    :param epoch: номер эпохи
    :param model: модель для обучения
    :param training_data_loader: тренировочный DataLoader
    :param criterion: функция потерь
    :param optimizer: оптимизатор
    :param device: cuda или cpu
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
         criterion: nn.Module,
         device: str):
    """
    функция для теста модели на отложенной выборке
    :param model: модель для теста
    :param testing_data_loader: тестовый DataLoader
    :param criterion: функция потерь
    :param device: cuda или cpu
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
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    model_out_path = os.path.join(history_dir, "model_epoch_{}.pth".format(epoch))
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
