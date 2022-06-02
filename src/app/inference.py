import os
import mlflow
import numpy as np
from PIL.Image import Resampling
from PIL import Image
from torchvision.transforms import ToTensor
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse

load_dotenv()

app = FastAPI()

os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MLFLOW_S3_ENDPOINT_URL')
print(os.getenv('MLFLOW_S3_ENDPOINT_URL'))


class Model:
    def __init__(self, model_name, model_stage):
        self.model = mlflow.pyfunc.load_model(f'models:/{model_name}/{model_stage}')

    def predict(self, data):
        return self.model.predict(data)


model = Model('cnn_YCbCr', 'Staging')


def super_resolve(input_filename: str, model):
    """
    функция для увеличения разрешения изображения
    :param input_filename: имя файла
    :param model: обученная модель
    :return:
    """
    img = Image.open(input_filename).convert('YCbCr')
    y, cb, cr = img.split()

    img_to_tensor = ToTensor()
    input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

    out = model.predict(input.detach().numpy())
    out_img_y = out[0]
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    # модель обучена только для яркости, поэтому остальные компоненты
    # проходят через интерполяцию.
    # реализация для RGB представлена в ноутбуке
    out_img_cb = cb.resize(out_img_y.size, Resampling.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Resampling.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

    return out_img


@app.post('/invocations')
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    with open(file.filename, 'wb') as f:
        f.write(contents)
    result = super_resolve(file.filename, model)
    os.remove(file.filename)
    result.save('out.jpg')
    return FileResponse('out.jpg')


@app.post("/images")
async def test_upload_file(file: UploadFile = File(...)):
    # file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()  # <-- Important!

    # example of how you can save the file
    with open(file.filename, "wb") as f:
        f.write(contents)

    return {"filename": file.filename, 'model': model.model}


@app.get("/")
async def main():
    content = """
<body>
<form action="/images" enctype="multipart/form-data" method="post">
<input name="file" type="file">
<input type="submit">
</form>
<form action="/invocations" enctype="multipart/form-data" method="post">
<input name="file" type="file">
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)


if (os.getenv('AWS_ACCESS_KEY_ID') is None) or (os.getenv('AWS_SECRET_ACCESS_KEY') is None):
    exit(1)
