FROM python:3.8

ENV PYTHONUNBUFFERED 1
ENV LANG en_US.utf8
ENV PYTHONIOENCODING utf-8

RUN mkdir /code
WORKDIR /code

ADD requirements.txt /code
ADD . /code/

RUN apt-get update
RUN pip install --upgrade pip
RUN pip install -r requirements.txt