FROM python:3.9.16-slim-buster

WORKDIR /api

COPY api_requirements.txt api_requirements.txt
RUN pip install --upgrade pip
RUN pip install -r api_requirements.txt

RUN python -m nltk.downloader stopwords

COPY . .

CMD ["python", "main.py"]