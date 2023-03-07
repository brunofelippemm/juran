FROM andreper/jupyterlab:3.0.0-spark-3.0.0
COPY . /app
RUN pip install -r requirements.txt
CMD jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=
