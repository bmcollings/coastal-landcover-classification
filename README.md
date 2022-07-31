# coastal-landcover-classification

This repository contains code that was used to develop the first national scale coastal specific landcover classification for New Zealand using public earth observation satellite data. [Google Earth Engine](https://earthengine.google.com) was used to develop composite imagery for 2019 from Sentinel-1 and Sentinel-2 sensors which was classified into nine landcover types using rule-based and supervised machine learning techniques in a Python workflow with [RSGISLib](http://rsgislib.org). 

| Classes |
| ------- |
| Artificial surfaces |
| Bare rock |
| Dark sand |
| Gravel |
| Intertidal |
| Light sand |
| Supratidal sand |
| Vegetation |
| Water | 

# Table of contents
1. [Installation](#installation)
2. [Python modules](#python)
3. [Jupyter notebooks](#notebooks)
4. [New Zealand coastal classification](#classification)

## Installation <a name='installation'></a>
Packages and dependencies handled are handled by conda

`conda create --name coastal-classification`

`conda activate coastal-classification`

`conda install --file requirements.txt`

`python -m ipykernel install --user --name=coastal-classification`

`jupyter notebook`

## Python code <a name='python'></a>
Code in this repository is contained in the ```coastal_landcover_classification``` package, which consists of two modules:

1. [```coastal_landcover_classification.composite```](https://github.com/bmcollings/coastal-landcover-classification/blob/main/coastal_landcover_classification/composite.py) handles the preprocessing and generation of annual composite imagery from all available images within a specified year. 
- Filters imagery to area of interest and year.
- Applies preprocessing steps to both optical and SAR data.
- Derives statistical aggregations of vegetation and water based indices (NDVI, NDWI, MNDWI and AWEI).
- Downloads composite imagery locally or to Google Drive.

2. [```coastal_landcover_classification.classification```](https://github.com/bmcollings/coastal-landcover-classification/blob/main/coastal_landcover_classification/classification.py) contains the functions to classify composite imagery to provide an annual coastal specific landcover classfication. 
- Applies a series of hierarchal rules using automated Otsu thresholding to identify water, intertidal and vegetation from multispectral composite imagery. 
- Classifies remaining classes using a random forest machine learning classifier trained with a manually derived national training dataset, included in this repository, using the multi-spectral and SAR composites images.

## Jupyter notebooks <a name='notebooks'></a>
A series of jupyter notebooks containing a working example of both steps are provided:

- [Annual composite creation](https://github.com/bmcollings/coastal-landcover-classification/blob/main/Notebooks/composite-development-example.ipynb)
- [Classification example](https://github.com/bmcollings/coastal-landcover-classification/blob/main/Notebooks/classification.ipynb)

## New Zealand coastal classification <a name='classification'></a>
The classification output generated for the year 2019 is available to view as a [Google Earth Engine App](https://bcol845.users.earthengine.app/view/nzcc-2019).
