FROM python:3.11.11

WORKDIR /Image_Compressor

COPY . .

RUN pip install -r req.txt

EXPOSE 5000

CMD ["python","app.py"]