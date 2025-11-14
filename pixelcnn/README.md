# Autoregressive Image Modeling

This directory contains code related to the following notebook:

https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial12/Autoregressive_Image_Modeling.html

## Set up

To run the code, first log in to Baskerville and join a compute node:

```sh
$ srun --qos turing --account usjs9456-ati-test --time 4:00:00 --nodes 1 \
    --gpus 1 --cpus-per-gpu 36 --mem 16384 --pty /bin/bash
```

Next load modules and deploy the virtual environment.
This will also install all of the required packages.

```sh
$ . ./setup_environment.sh
```

## Executing the code

To execute the initial part of the nobebook:

```sh
$ python pixelcnn01.py
```

To execute the second part of the notebook:


```sh
$ python pixelcnn02.py
```

## Outputs

The figures will be output to the `figures` directory, but will also be output to the screen as ASCII art.

To display one of the PNG figures as ASCII art, the `display.py` utility can be used.
Just pass the name of the image file to display:

```sh
$ python display.py figures/figure1.png
```
