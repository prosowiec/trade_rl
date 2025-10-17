FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get update && apt-get install -y ffmpeg libsm6 libxext6

COPY . .

EXPOSE 8501

CMD ["bash", "-c", "python main.py & streamlit run dashboard.py"]
