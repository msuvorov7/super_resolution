import os
import argparse
import sys

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

from src.features.build_dataset import get_training_set, get_test_set
from src.models.model import SRCNN
from src.visualization.visualize import plot_psnr, plot_loss


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
    avg_psnr = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        psnr = 10 * np.log10(1 / loss.item())
        avg_psnr += psnr
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    return epoch_loss / len(training_data_loader), avg_psnr / len(training_data_loader)


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
    epoch_loss = 0
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            epoch_loss += mse.item()
            psnr = 10 * np.log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    return epoch_loss / len(testing_data_loader), avg_psnr / len(testing_data_loader)


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


def fit(epochs: int):
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

    train_losses = []
    val_losses = []
    train_psnrs = []
    val_psnrs = []

    for epoch in range(1, epochs):
        train_loss, train_psnr = train(epoch, model, training_data_loader, criterion, optimizer, device)
        val_loss, val_psnr = test(model, testing_data_loader, criterion, device)
        checkpoint(epoch, model, 'models')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_psnrs.append(train_psnr)
        val_psnrs.append(val_psnr)

    torch.save(model, 'models/model.pth')

    plot_psnr(train_psnrs, val_psnrs)
    plot_loss(train_losses, val_losses)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--epochs', dest='epochs', type=int, required=True)
    args = arg_parser.parse_args()

    fit(args.epochs)
