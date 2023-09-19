FROM python:3.11

COPY . /app

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

ENTRYPOINT ["bash", "-c", "export PYTHONPATH=$PYTHONPATH:$PWD && streamlit run src/data/dashboard.py"]