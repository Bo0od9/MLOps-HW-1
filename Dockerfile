FROM python:3.12-slim

WORKDIR /app

# создаем нужные паки
RUN mkdir -p /app/logs /app/input /app/output \
 && touch /app/logs/service.log \
 && chmod -R 777 /app/logs /app/input /app/output

# установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Точки монтирования
VOLUME /app/input
VOLUME /app/output

# запуск
CMD ["python", "-u", "app/app.py"]
