FROM poldracklab/fmriprep:1.4.1

RUN conda create --name fmriprep_env --clone base

COPY requirements.txt /root/src/req/requirements.txt
RUN pip install -r /root/src/req/requirements.txt

COPY . /root/src/cpr
RUN cd /root/src/cpr && \
    pip install -e .

ENTRYPOINT ["/usr/local/miniconda/bin/run_cpr.py"]

