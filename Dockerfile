FROM pytorch-20.08:latest
RUN pip install soundfile
RUN pip install torchsummary
WORKDIR /code


