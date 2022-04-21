import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.features.build_dataset import get_training_set, get_test_set
from src.models.model import SRCNN
from src.models.predict_model import super_resolve
from src.models.train_model import test, checkpoint, train


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

    for epoch in range(1, 7):
        train(epoch, model, training_data_loader, criterion, optimizer, device)
        test(model, testing_data_loader, criterion, device)
        checkpoint(epoch, model, 'models')

    super_resolve('test.jpg', 'out.jpg', model)
