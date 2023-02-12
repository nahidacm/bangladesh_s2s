FROM continuumio/anaconda3

WORKDIR /app

RUN apt-get update && apt-get install -y libgtk2.0-dev && \
    rm -rf /var/lib/apt/lists/*
RUN /opt/conda/bin/conda update -n base -c defaults conda && \
    /opt/conda/bin/conda install python=3.8
RUN /opt/conda/bin/conda install anaconda-client
RUN /opt/conda/bin/conda install --channel https://conda.anaconda.org/menpo \ 
    datetime numpy xarray \ 
    matplotlib mpl_toolkits cartopy \ 
    requests pandas geopandas  \ 
    configparser xcast cmocean docxptl -y

# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "run.py"]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "download_ecmwf_s2s_from_wi_api.py"]
