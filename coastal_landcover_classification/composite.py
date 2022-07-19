#!/usr/bin/env python

# This script contains the functions to generate composite imagery from 
# Sentinel and Landsat datasets using Google Earth Engine
# for the classification of coastal zones in New Zealand.
# Options to export to google drive. 

# import modules
from cmath import nan
from tkinter.font import names
import rsgislib
import rsgislib.imageutils
# import required modules
import os
import ee
import glob
import shutil
from osgeo import gdal
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from functools import reduce
import time
import json
import tqdm
import math
import io 

# Initialize GEE
ee.Initialize()

# define global variables
# dict containing sensor image bands 
img_bands = {'S2': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7','B8', 'B8A', 'B11', 'B12'],
        'LS7': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7'],
        'LS7_sr': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
        'LS8': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
        'LS8_sr': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
        'S1': ['HH', 'HV', 'VV', 'VH', 'angle']}

# dict containing sensor GEE snippets for optical collections store SR and TOA collections as list
sensor_id = {'S2': ['COPERNICUS/S2_SR', 'COPERNICUS/S2'],
        'LS7': ['LANDSAT/LE07/C02/T1_L2', 'LANDSAT/LE07/C02/T1_TOA'],
        'LS8': ['LANDSAT/LC08/C02/T1_L2', 'LANDSAT/LC08/C02/T1_TOA'],
        'S1': ['COPERNICUS/S1_GRD']} 

# list of band names
band_names = ['blue', 'green', 'red', 'RE1', 'RE2', 'RE3', 'NIR', 'RE4', 'SWIR1', 'SWIR2']


def convert_img(inputFolder, outputFolder, inFormat, outFormat='KEA'):
    """
    function to convert all gdal supported image files in folder to specficed gdal_format format using gdal_translate

    Args
    inputFolder - inputFolder containing images to be converted
    outputFolder - destination folder for .kea images
    inFormat - input file format
    outFormat - output format default = 'KEA'
    
    """
    # navigate to inputFolder
    #os.chdir(inputFolder)
    # create outputFolder if it doesn't exist
    try:
        os.mkdir(outputFolder) 
    except:
        os.path.exists(outputFolder) == True
    # return list of images
    imglist = []
    for f in glob.glob(inputFolder + '/*.' + inFormat):
        print(f)
        imglist.append(f)
    # convert using gdal_translate
    for f in tqdm.tqdm(imglist):
        # get band names
        bandNames = rsgislib.imageutils.get_band_names(f)
        gdal_translate = 'gdal_translate -of ' + inFormat+ ' ' + f + ' ' + f[:-4] + '.' + outFormat
        print(gdal_translate)
        os.system(gdal_translate)
        # input band names and move outputs to outputFolder
        for f in glob.glob(inputFolder + '/*.' + outFormat):
            rsgislib.imageutils.set_band_names(f, bandNames)
            shutil.move(f, outputFolder)


def set_nodata_val(image, no_data_val):
    """
    function to set no data value for image
    
    Arg
    image - str, filepath to image requiring no_data value
    no_data_val - int, no data value to set
    """
    ds = gdal.OpenEx(image, gdal.GA_Update)
    for i in range(ds.RasterCount):
        ds.GetRasterBand(i + 1).SetNoDataValue(no_data_val)


def mosaic(inputImageList, nodataVal=-99, outputFormat='KEA', dataType=rsgislib.TYPE_32FLOAT):
    """
    function to moasaic group of images in specified folder and populate image with statistics

    Args

    inputImageList - list of images to be mosaiced
    outputImage - str, output mosaic
    inputFormat - str, gdal format of input images default=kea
    nodataVal - float, output no data value
    innodataVal - float, no data value for input images
    band - int, input image band to get no data value from
    outputFormat - str, containing output image format default=KEA
    datatype - rsgislib datatype default=rsgislib.TYPE_32FLOAT
    inputFN - str, if images to be merged are in different subdirectories with same filename default=None 
    
    Returns

    outputImage - mosaiced image
    """
    
    # get band names
    bandNames = rsgislib.imageutils.get_band_names(inputImageList[0])
    print(bandNames)
    outputImage = inputImageList[0][:2] + '-composite.kea'
    # define reamaining args
    overlapBehaviour = 0   
    skipBand = 1
    # mosaic with imageutils.createImageMosaic and populate image stats
    innodataVal = nodataVal
    rsgislib.imageutils.create_img_mosaic(inputImageList, outputImage, nodataVal, innodataVal, skipBand, overlapBehaviour, outputFormat, dataType)
    rsgislib.imageutils.set_band_names(outputImage, bandNames)
    set_nodata_val(outputImage, nodataVal)
    


def rename_img_bands(sensor):
    """function to rename optical image bands for ee.Image in ee.ImageCollection when using .map function
    
    Args
    sensor -  string sensor type that bands are being renamed
    
    returns 
    ee.Image with renamed bands
    """

    def rename(image):
        bands = img_bands[sensor]
        names = []
        if sensor == 'S2':
            names = band_names
        else:
            names = band_names[:3] + band_names[6:7] + band_names[-2:]
        
        return image.select(bands).rename(names)
    
    return(rename)


def resample_image(image, crs='EPSG: 2193', pixel_size=20):
    """
    function to resample ee.image object 
    
    Args
    image - ee.image object to be resampled
    crs - reference system as epsg code
    pixel_size - resampled pixel size"""

    bands = image.bandNames()
    resampled_bands = image.select(bands).reproject({'crs': crs, 'scale': pixel_size})
    return resampled_bands


def mask_clouds_S2_QA60(image):
    """function to mask Sentinel-2 ee.Image object using QA60 band

    Args
    image = ee.Image object

    Returns
    ee.image object with updated cloud mask
    """

    qa = image.select('QA60')
    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    # clear if both flags set to zero.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0) \
        .And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask)


def mask_clouds_S2_probablity(image):
    """
    function to mask sentinel-2 ee.image object using cloud probability band
    
    Returns
    Sentinel-2 ee.image object with updated cloud mask
    """

    clouds = ee.Image(image.get('cloud_mask')).select('probability')
    isNotCloud = clouds.lt(65)
    return image.updateMask(isNotCloud)


def mask_clouds_LS_qa(image):
    """
    function to mask Landsat ee.image object using QA_pixel band from Fmask
    
    Args
    landsat ee.image object
    
    Returns
    landsat ee.image object with cloud masked
    """
    # define bit_masks
    shadow_bit_mask = (1 << 4)
    cloud_bit_mask = (1 << 3)
    dcloudBitMask = (1 << 1)
    # get qa image band
    qa = image.select('QA_PIXEL')

    # define mask
    mask = qa.bitwiseAnd(shadow_bit_mask).eq(0) \
        .And(qa.bitwiseAnd(cloud_bit_mask).eq(0)) \
        .And(qa.bitwiseAnd(dcloudBitMask).eq(0))
    
    return image.updateMask(mask)


def apply_scale_factors(image):
    """
    function to apply scale factors to landsat SR collection 2 image
    """

    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)

    return image.addBands(opticalBands, None, True) \
              .addBands(thermalBands, None, True)

def apply_ndvi(image):
    """function to calculate ndvi for ee.image object and add band to object
    
    Args 
    image - ee.image object

    returns
    ee.image object with ndvi  band
    """
    calc_ndvi = image.normalizedDifference(['NIR', 'red'])
    ndvi = calc_ndvi.select([0], ['ndvi'])
    
    return ee.Image.cat([image, ndvi])


def apply_ndwi(image):
    """function to calculate ndwi for ee.image object and add band to object
    
    Args 
    image - ee.image object

    returns
    ee.image object with ndwi  band
    """
    calc_ndwi = image.normalizedDifference(['green', 'NIR'])
    ndwi = calc_ndwi.select([0], ['ndwi'])
    
    return ee.Image.cat([image, ndwi])

def apply_mndwi(image):
    """function to calculate mndwi for ee.image object and add band to object
    
    Args 
    image - ee.image object

    returns
    ee.image object with mndwi  band
    """
    calc_mndwi = image.normalizedDifference(['green', 'SWIR1'])
    mndwi = calc_mndwi.select([0], ['mndwi'])
    
    return ee.Image.cat([image, mndwi])


def apply_awei(image):
    """function to calculate ndwi for ee.image object and add band to object
    
    Args 
    image - ee.image object

    returns
    ee.image object with ndwi  band
    """
    calc_awei = image.expression("4*(b('green')-b('SWIR1'))-(0.25*b('NIR')+2.75*b('SWIR2'))")
    awei = calc_awei.select([0], ['awei'])
    
    return ee.Image.cat([image, awei])


def mask_angle(angle, greater_than=True):
    """
    function to filter edge of sar image based on incidence angle
    Args 
    angle - float specifying incidence angle
    greater_than - boolean, if True will mask angles less than angle specified, if False will mask values greater than angle specified """

    def apply_mask(image): 
        angle_band = image.select(['angle'])
        if greater_than == True:
            return image.updateMask(angle_band.gt(angle))
        if greater_than == False:
            return image.updateMask(angle_band.lt(angle))
    
    return(apply_mask)


def calc_sar_nd(image):
    return image.addBands(image.expression('(VH - VV)/(VH + VV)', 
    {'VH': image.select(['VH']),
    'VV': image.select(['VV'])
    }))


def shp_to_featureCollection(shapefile):
    """
    function to read a shapefile as a ee.featureCollection using geopandas
    
    Args
    shapefile - path to shapefile to be read as featureCollection GEE object
    
    Returns
    ee.FeatureCollection object"""
    # read shapefile as gdf and convert to json_dict
    gdf = gpd.read_file(shapefile)
    if gdf.geom_type[0] == 'LineString':
        #gdf = gdf.buffer(1500)
        gdf = gdf.to_crs(4326)
    else:
        gdf = gdf.to_crs(4326)
    geo_json = gdf.to_json()
    json_dict = json.loads(geo_json)
    features = []
    # iterate over json features convert ee.Geometries and read as ee.Feature
    for feature in json_dict['features']:
        # derive ee.Geometry type from json_dict
        if feature['geometry']['type'] == 'LineString':
            line = ee.Feature(ee.Geometry.LineString(feature['geometry']['coordinates']))
            features.append(line.buffer(1500))
        if feature['geometry']['type'] == 'Polygon':
            features.append(ee.Feature(ee.Geometry.Polygon(feature['geometry']['coordinates'])))
        if feature['geometry']['type'] == 'Point':
            features.append(ee.Feature(ee.Geometry.Point(feature['geometry']['coordinates'])))


    return ee.FeatureCollection(features)


def buffer(buffer):
    """
    function to buffer features in featureCollection
    Args 
    featureCollection - ee.object to be buffered 
    buffer - buffer distance in metres

    returns
    buffered ee.featureCollection
    """
    def apply_buffer(feature):
        return feature.buffer(buffer)
    
    return apply_buffer


def run_task(task, mins):
    """
    function to run ee.batch.export and check status of task every number of minutes specified by mins
    """

    secs = mins * 60
    task.start()
    while task.active():
        print(task.status())
        time.sleep(secs)
    

def create_optical_composite(year, region_shp, sensor, crs, pixel_size, save_outputs_local=None, use_toa=True, down_folder=None, file_name_prefix=None):
    """
    function to generate and download annual composite imagery from GEE for pixel-based coastal classification 
    and change detection
    
    Args
    year - year as integer eg. 2019
    sensor - sensor type to build composite image as string (S2, LS7, LS8)
    region - shapefile defining region for composite, accepts polygons and lines, if polyline representing coastline output will be coast
            zone defined as 3km buffer zone around coastline
    crs - the output coordinate reference system as epsg code
    pixel_size - output resolution of composite image
    save_output_local - if region is < Xkm2 outputs can be saved locally by specifying the local file path otherwise default is 
                        and outputs will be saved to users google drive account
    use_toa - default = True, will use TOA image collections to generate water based remote sensing indices
    save_local - if region is small specify a local folder to save output to default = None and output will saved to google drive 
    down_folder - string stating name of google drive to download image to 
    file_name_prefix - string, if working with multiple locations specify a three character prefix e.g. 'AUK' 
    
    returns
    composite image as a GeoTiff to specified local folder or to google drive
    """
    
    # define date ranges 
    start_date = str(year) + '-01-01'
    end_date = str(year + 1) + '-01-01'

    # convert region to ee.featureCollection 
    roi = shp_to_featureCollection(region_shp)

    # define sr image collection
    sr_image_collection = ee.ImageCollection(sensor_id[sensor][0]) \
        .filterBounds(roi) \
        .filterDate(start_date, end_date)
    # define toa collection 
    if use_toa == True:
        toa_image_collection = ee.ImageCollection(sensor_id[sensor][1]) \
            .filterBounds(roi) \
            .filterDate(start_date, end_date)
    else:
        sr_image_collection = toa_image_collection
    
    if sensor == 'S2' and year <= 2017:
        sr_image_collection = toa_image_collection
    
    # perform cloud masking based on sensor type
    if sensor == 'S2':
        s2_cloud_prob = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
            .filterBounds(roi) \
            .filterDate(start_date, end_date)
        # join cloud_prob with image_collections
        sr_image_collection = ee.Join.saveFirst('cloud_mask').apply(sr_image_collection, s2_cloud_prob,
                ee.Filter.equals(leftField = 'system:index', rightField = 'system:index'))
        if use_toa == True:
            toa_image_collection = ee.Join.saveFirst('cloud_mask').apply(toa_image_collection, 
            s2_cloud_prob, 
            ee.Filter.equals(leftField = 'system:index', rightField = 'system:index'))
        # mask clouds 
        toa_image_collection_cm = ee.ImageCollection(toa_image_collection) \
            .map(mask_clouds_S2_probablity) \
            .map(mask_clouds_S2_QA60) \
            .map(rename_img_bands(sensor))
        sr_image_collection_cm = ee.ImageCollection(sr_image_collection) \
            .map(mask_clouds_S2_probablity) \
            .map(mask_clouds_S2_QA60) \
            .map(rename_img_bands(sensor))        
    
    if sensor == 'LS7':
        # LS_sr bands required 
        sr_image_collection_cm = ee.ImageCollection(sr_image_collection) \
            .map(mask_clouds_LS_qa) \
            .map(apply_scale_factors) \
            .map(rename_img_bands('LS7_sr'))
        if use_toa == True:
            toa_image_collection_cm = ee.ImageCollection(toa_image_collection) \
                .map(mask_clouds_LS_qa) \
                .map(rename_img_bands(sensor))  
        else: 
            toa_image_collection_cm = ee.ImageCollection(toa_image_collection) \
                .map(mask_clouds_LS_qa) \
                .map(apply_scale_factors) \
                .map(rename_img_bands('LS7_sr')) 
    
    if sensor == 'LS8':
        # LS_sr bands required 
        sr_image_collection_cm = ee.ImageCollection(sr_image_collection) \
            .map(mask_clouds_LS_qa) \
            .map(apply_scale_factors) \
            .map(rename_img_bands('LS8_sr'))
        if use_toa == True:
            toa_image_collection_cm = ee.ImageCollection(toa_image_collection) \
                .map(mask_clouds_LS_qa) \
                .map(rename_img_bands(sensor))  
        else: 
            toa_image_collection_cm = ee.ImageCollection(toa_image_collection) \
                .map(mask_clouds_LS_qa) \
                .map(apply_scale_factors) \
                .map(rename_img_bands('LS8_sr')) 

    # generate indices for image_collections
    toa_image_collection_indices = ee.ImageCollection(toa_image_collection_cm) \
        .map(apply_ndwi) \
        .map(apply_mndwi) \
        .map(apply_awei)
    sr_image_collection_indices = ee.ImageCollection(sr_image_collection_cm) \
        .map(apply_ndvi) 
    
    # reduce sr_collection to generate optical composite bands
    # select bands names for S2 or LS
    print(sensor)
    if sensor == 'S2':
        b_names = band_names
    if sensor == 'LS7':
        b_names =  band_names[:3] + band_names[6:7] + band_names[8:9]
    if sensor == 'LS8':
        b_names =  band_names[:3] + band_names[6:7] + band_names[8:10]
    print(b_names)
    # derive p15 composite bands for optical bands
    image_bands = sr_image_collection_indices.select(b_names)
    p15_image = image_bands.reduce(ee.Reducer.percentile([15]))
    # reduce ndvi from sr_collection
    ndvi = sr_image_collection_indices.select(['ndvi'])
    ndvi_composite = ndvi.reduce(ee.Reducer.intervalMean(10,90).setOutputs(['intMn1090']))
    # reduce water indices from toa_collection
    indices = toa_image_collection_indices.select(['ndwi', 'mndwi', 'awei'])
    reducers = ee.Reducer.minMax() \
        .combine(ee.Reducer.median(), '', True) \
        .combine(ee.Reducer.stdDev(), '', True) \
        .combine(ee.Reducer.percentile([10,25,50,75,90]), '', True)
    indices_composites = indices.reduce(reducers)

    # add composite bands to single ee.image object
    composite_image = p15_image \
        .addBands(ndvi_composite) \
        .addBands(indices_composites)
    
    # clip composite_image for export
    clip = composite_image.clip(roi)
    
    # export composite image to google drive if local folder not specified
    # if down_folder is None set google drive dir name to year 
    if down_folder == None:
        dir =  str(year)
    else:
        dir = down_folder

    task = ee.batch.Export.image.toDrive(image=clip.toFloat(),
        region=roi.geometry(),
        description='export_' + sensor + '_' + str(year),
        folder=dir,
        fileNamePrefix=sensor + '-' + str(year),
        scale=pixel_size,
        crs=crs,
        maxPixels=1e13,
        skipEmptyTiles=True
    )
    
    # run download task
    run_task(task, 2)

def create_sar_composite(year, region_shp, sensor, crs, pixel_size, save_outputs_local=None, down_folder=None):
    """
    function to generate and download annual sar composite imagery from GEE for pixel-based coastal classification 
    and change detection
    
    Args
    year - year as integer eg. 2019
    sensor -  sar sensor type to build composite image as string (currently configured for Sentinel-1 GRD only)
    region - shapefile defining region for composite, accepts polygons and lines, if polyline representing coastline output will be coast
            zone defined as 3km buffer zone around coastline
    crs - the output coordinate reference system as epsg code
    pixel_size - output resolution of composite image
    save_output_local - if region is < Xkm2 outputs can be saved locally by specifying the local file path otherwise default is None
                        and outputs will be saved to users google drive accoun
    down_folder - string stating name of google drive to download image to if None will name folder by year 
    
    returns
    composite image as a GeoTiff to specified local folder or to google drive
    """
    # define date ranges 
    start_date = str(year) + '-01-01'
    end_date = str(year + 1) + '-01-01'

    # convert region to ee.featureCollection 
    roi = shp_to_featureCollection(region_shp)

    # define sr image collection
    sar_image_collection = ee.ImageCollection(sensor_id[sensor][0]) \
        .filterBounds(roi) \
        .filterDate(start_date, end_date)
    
    # filter by polarisation 
    vh = sar_image_collection.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    
    # mask incidence angles below 30 and above 45 degrees
    vh = vh.map(mask_angle(30.63993, True))
    vh = vh.map(mask_angle(45.53993, False))

    # claculate ratio between VV/VH
    vh = vh.map(calc_sar_nd)

    # filter by ascending and descending orbits
    vh_descending = vh.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    vh_ascending = vh.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))

    # calc mean reduction on collections
    desc_mean = vh_descending.mean()
    asc_mean = vh_ascending.mean()

    # define band_names for composites
    desc_bandNames = ['VV_desc', 'VH_desc', 'VVVH_desc']
    asc_bandNames = ['VV_asc', 'VH_asc', 'VVVH_asc' ]

    # rename bands and merge for export
    desc_mean = desc_mean.select('VV', 'VH', 'VH_1').rename(desc_bandNames)
    asc_mean = asc_mean.select('VV', 'VH', 'VH_1').rename(asc_bandNames)

    composite = desc_mean.addBands(asc_mean)

    # clip composite_image for export
    clip = composite.clip(roi)

    # export composite image to google drive if local folder not specified
    # if down_folder is None set google drive dir name to year 
    if down_folder == None:
        dir =  str(year)
    else:
        dir = down_folder
    
    if file_name_prefix == None:
        file_name = sensor + '-' + str(year)
    else:
        file_name = file_name_prefix + '-' + sensor + '-' + str(year)

    task = ee.batch.Export.image.toDrive(image=clip.toFloat(),
        region=roi.geometry(),
        description='export_' + sensor + '_' + str(year),
        folder=dir,
        fileNamePrefix= file_name,
        scale=pixel_size,
        crs=crs,
        maxPixels=1e13,
        skipEmptyTiles=True
    )
    
    # run download task
    run_task(task, 2)

    

    
