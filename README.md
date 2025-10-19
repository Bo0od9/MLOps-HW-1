# Fraud Detection Service (MlOps and ML System Design HW 1)

## Структура проекта

```
HW-1/
├── app/
│   └── app.py              # Сервис: следит за input/, вызывает препроцессинг и инференс модели
├── src/
│   ├── preprocess.py       # Препроцессинг данных
│   └── scorer.py           # Загрузка модели и инференс предсказаний
├── configs/
│   └── config.yaml         # Конфиг
├── models/
│   └── model.cbm           # Обученная CatBoost модель
├── input/                  # сюда класть test.csv
├── output/                 # сюда сохраняются все результаты
├── notebooks/
│   ├── eda_and_train.ipynb # ноутбук для обучения модели
│   └── /data
├── logs/                   # Логи
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Что делает сервис

* Следит за папкой `input/`
* Как только появляется файл `test.csv` — сервис запускает:

  1. `preprocess_data()` — обработка данных
  2. `make_pred()` — загрузка модели, предсказания
  3. Сохранение `sample_submission.csv`, файла с топ-5 и график плотности распределения предсказанных моделью скоров в `output/`
* Работает на CPU, **без обучения модели**, только inference

---

## Запуск через Docker

### Собрать образ

```bash
docker build -t fraud_service .
```

### Запустить контейнер

```bash
docker run -it --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/logs:/app/logs" \
  fraud_service
```

### Положить файл для предсказания

```bash
cp some_file.csv input/test.csv
```

### Проверить результаты

* `output/sample_submission.csv` — предсказания
* `output/feature_importances.json` — топ-5 фич
* `output/scores_density.png` — плотность вероятностей

---

## Требования к файлу test.csv

* Должны быть **те же колонки**, что в соревновании [Kaggle](https://www.kaggle.com/competitions/teta-ml-1-2025/overview)
* Формат CSV, разделитель запятая
* Можно взять файл из `/notebooks/data`

---

## Возможные ошибки

| Проблема               | Решение                                    |
| ---------------------- | ------------------------------------------ |
| `Model file not found` | Проверь, что `models/model.cbm` существует |
| `Matplotlib GUI error` | Установить `MPLBACKEND=Agg`                |
| Нет вывода             | Проверь `logs/service.log`                 |
