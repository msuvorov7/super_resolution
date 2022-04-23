from src.data.data_load import download_bsd300
from src.models.predict_model import super_resolve
from src.models.train_model import fit


if __name__ == '__main__':
    download_bsd300()
    fit(epochs=20)
    super_resolve('test.jpg', 'out.jpg', 'models/model.py')
