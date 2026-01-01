# AMD64 ile uyumlu, stabil Python imajı
FROM python:3.10-slim

# Ortam değişkenleri (log + buffer sorunlarını önler)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Çalışma dizini
WORKDIR /app

# Sistem bağımlılıkları (ML + matplotlib için GEREKLİ)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Requirements önce (cache için önemli)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Proje dosyaları
COPY train_cleaned_data.py .
COPY telco_cleaned.csv .
COPY config ./config

# Plot klasörü (runtime'da hata almamak için)
RUN mkdir -p plots

# Varsayılan komut
CMD ["python", "train_cleaned_data.py"]
