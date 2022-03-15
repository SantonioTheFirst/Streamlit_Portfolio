# FROM ubuntu:latest

# RUN apt-get update && apt-get upgrade -y

# RUN apt-get install python3.8 python3-pip python-dev -y

FROM python:3.8.10-slim

WORKDIR /usr/src/app

COPY . /usr/src/app/

RUN pip install --upgrade pip

# RUN python -m venv venv

# RUN source venv/bin/activate

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]