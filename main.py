from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pydantic import TypeAdapter
from typing import List
import json
import pickle
import pandas as pd
from custom_transformer import MyTransformer


model = pickle.load(open('model.pkl', 'rb'))
app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    d = dict(item)
    d.pop('selling_price', None)
    return model.predict(pd.DataFrame({k: [v] for k, v in d.items()}))


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    return [predict_item(i) for i in items]


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    file.file.close()
    df = df.dropna()
    df_json = json.loads(df.to_json(orient="records"))
    ta = TypeAdapter(List[Item])
    py_list = ta.validate_python(df_json)
    predicts = predict_items(py_list)
    predicts = [p[0] for p in predicts]
    result = [dict(d, predict=float(p)) for d, p in zip(df_json, predicts)]
    df = pd.json_normalize(result)
    df.to_csv(file.filename)
    return FileResponse(path=file.filename, filename=file.filename, media_type='text/csv')
