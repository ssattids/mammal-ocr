FROM ubuntu:22.04
# update apt-get
RUN apt-get update -y
# install tesseract
RUN apt-get install -y tesseract-ocr 
# install python
RUN apt-get install -y python3.10
RUN apt-get install -y python3-pip
RUN apt-get install -y git
# install pytesseract
RUN pip3 install pytesseract
RUN pip3 install fastapi
RUN pip3 install PyMuPDF
RUN pip3 install pillow
RUN pip3 install jupyter