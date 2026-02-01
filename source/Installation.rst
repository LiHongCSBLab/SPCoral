.. highlight:: shell

============
Installation
============


Install by PyPi
---------------

**Step 1:**

Prepare conda environment for SPCoral:
::

	conda create -n SPCoral python=3.9
	conda activate SPCoral

**Step 2:**

Install SOAPy using `pip`:
::

	pip install spcoral


Install by github
-----------------

Download the file from github:
::

    cd SPCoral
    python setup.py build
    python setup.py install


Requirements of SOAPy
-----------------

Those will be installed automatically when using pip.

::

    anndata==0.10.9
    dgl==1.0.2
    esda==2.5.1
    igraph==0.11.8
    networkx==3.2.1
    numpy==1.26.4
    pandas==2.2.2
    pysankey==0.0.1
    scanpy==1.10.3
    scipy==1.12.0
    seaborn==0.13.2
    torch==2.1.0
    matplotlib
    tqdm
    


