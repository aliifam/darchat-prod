# Gunakan image resmi Python dengan versi 3.10
FROM python:3.10

# Setel direktori kerja di dalam kontainer
WORKDIR /app

# Salin file requirements.txt ke kontainer
COPY ./requirements.txt /app/requirements.txt

# Instal dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Salin kode sumber lokal ke kontainer
COPY . /app

# Perintah untuk menjalankan main.py saat kontainer diluncurkan di background
CMD ["python", "main.py"]
