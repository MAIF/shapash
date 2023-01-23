FROM python:3.9-slim

RUN apt-get update
RUN apt-get install -y chromium chromium-driver 

RUN apt-get install -y libgomp1

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.dev.txt /app/

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r /app/requirements.dev.txt

COPY /shapash/. /app/shapash/
COPY /tests/. /app/tests/

CMD ["python", "-m", "pytest"]
