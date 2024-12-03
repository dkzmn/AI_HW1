import requests
import pandas as pd
from sklearn.metrics import r2_score

test_file = 'https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv'
url_item = 'http://127.0.0.1:8000/predict_item'
url_upload = 'http://127.0.0.1:8000/upload'

# Скачиваем тестовый файл
response = requests.get(test_file)
if response.status_code == 200:
    open("cars_test.csv", 'wb').write(response.content)

# Создаем датасет и построчно отправляем его в сервис на прогноз
df = pd.read_csv('cars_test.csv')
df = df.dropna()
y = df['selling_price']
count = 0
predict = []
for i in df.to_dict('records'):
    r = requests.post(url_item, json=i)
    predict.append(r.json())
    count += 1
# По всем прогнозам считаем скор
print(f'{count} items predicted with R2 score {r2_score(y, predict)}')

# Загружаем файл в сервис целиком
r = requests.post(url_upload, files={'file': open('cars_test.csv', 'rb')})
# Получаем от сервиса ответный файл с прогнозами
open('new_cars_test.csv', 'wb').write(r.content)
df = pd.read_csv('cars_test.csv')
df = df.dropna()
# Снова считаем скор прогнозов
print(f'All file  predicted with R2 score {r2_score(df['selling_price'], df['predict'])}')
