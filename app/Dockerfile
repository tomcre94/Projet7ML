# this sets up the container with python 3.10
FROM python:3.10-slim

# this copies everything in the Dockerfile directory to /app directory in the container
COPY . /app

# change working dir
WORKDIR /app

# installing the dependencies
RUN pip install -r requirements.txt

# exposing port 5000
EXPOSE 5000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# command to run
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.main:app"]