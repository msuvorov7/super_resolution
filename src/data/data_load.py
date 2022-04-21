import os
import tarfile
from six.moves import urllib


def download_bsd300(dest: str = 'data/raw') -> str:
    """
    загружает датасет, если его еще нет
    :param dest: имя папки для изображений
    :return: полный путь до изображений
    """
    output_image_dir = os.path.join(dest, 'BSDS300/images')

    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)
        file_path = os.path.join(dest, os.path.basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        os.remove(file_path)

    return output_image_dir
