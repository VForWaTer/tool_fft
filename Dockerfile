# Pull any base image that includes python3
FROM python:3.10

# install the toolbox runner tools
RUN pip install json2args

# install numpy and scipy
RUN pip install numpy==1.24.2 scipy==1.10.1 plotly==5.13.1 matplotlib==3.7.0


# create the tool input structure
RUN mkdir /in
COPY ./in /in
RUN mkdir /out
RUN mkdir /src
COPY ./src /src

WORKDIR /src
CMD ["python", "run.py"]
