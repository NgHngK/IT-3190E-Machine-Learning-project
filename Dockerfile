FROM python:3.10.12

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install required OS-level packages

# Install Python dependencies
COPY requirement.txt ./
#RUN pip install --upgrade pip
RUN pip install -r requirement.txt

# Copy all code
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
