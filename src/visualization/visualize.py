import matplotlib.pyplot as plt


def plot_loss(train_loss: list, val_loss: list):
    plt.figure(figsize=(16, 8))
    plt.plot(train_loss, marker='s', label='Train Loss')
    plt.plot(val_loss, marker='s', label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.savefig('mse_loss.jpg')


def plot_psnr(train_psnr: list, val_psnr: list):
    plt.figure(figsize=(16, 8))
    plt.plot(train_psnr, marker='s', label='Train PSNR')
    plt.plot(val_psnr, marker='s', label='Validation PSNR')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.savefig('psnr.jpg')
