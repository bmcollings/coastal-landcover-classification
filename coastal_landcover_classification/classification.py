#!/usr/bin/env python

# This script contains the classification functions for coastal landcover classification using RSGISlib (LINK)
#       * rule base functions that use otsu thresholding
#       * training functions to extract training from vector data
#       * machine learning functions to apply random forest algorithm 

# import libraries
# import rsgislib libraries
import rsgislib
from rsgislib import imagecalc
from rsgislib.imagecalc import BandDefn
from rsgislib import vectorutils
from rsgislib import imageutils
from rsgislib import rastergis
import rsgislib.classification
import rsgislib.classification.classsklearn

# import other modules
from rios import rat
import numpy as np
from skimage import filters
from osgeo import gdal
import tqdm
import multiprocessing
from functools import partial
import glob
import os
import shutil
import random
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import uuid
from coastal_landcover_classification.composite import set_nodata_val



def classify_otsu(inputImage, imgBand, outputImage, gdalformat='KEA'):
    """
    Function that returns an mask/binary classification image based on Otsu threshold for use in remote sensing classification

    Args
    inputImage - str, image containing band that otsu value will be generated from
    imgBand -  int, band number
    gdalformat - default is 'KEA'

    Returns
    outputImage - binary image where 0 = non classified pixels and 1 = classified pixels
    """

    # get band name from image
    bands = imageutils.get_band_names(inputImage)
    count = 0
    for name in bands:
        count += 1
        if count == imgBand:
            bandName = name
            print(name)
    raster = gdal.Open(inputImage)
    array = np.array(raster.GetRasterBand(imgBand).ReadAsArray())
    # remove nan and nodatavals
    array = array[~np.isnan(array)]
    array = array[array != -99]
    # define band info for image mask
    bandDefns = []
    bandDefns.append(rsgislib.imagecalc.BandDefn(bandName, inputImage, imgBand))
    # calculate otsu
    threshold = filters.threshold_otsu(array)
    print(str(threshold))
    # define expression and classify image
    exp = '(' + bandName + '<' + str(threshold) + ')?1: (' + bandName + '== -99)?-99:0'
    print(exp)
    print(outputImage)
    print(str(bandDefns))
    rsgislib.imagecalc.band_math(outputImage, exp, gdalformat, rsgislib.TYPE_8UINT, bandDefns)
    raster = None


def generate_elevation_mask(inputImage, imgBand, outputImage, eleThreshold, bandName=None, gdalformat='KEA'):
	"""
	function to generate a binary elevation mask where 0 = above threshold value and 1 = below threshold value

	Args
	inputImage - str, filepath to image containing DEM
	imgBand - band number containing DEM
	outputImage - str, filepath to output elevation mask
	eleThreshold - int, elevation threshold
	bandName - str, name band if default is band 1 to avoid error default=None 
	gdalformat - output image format default=KEA
    

	Returns
	Image - elevation mask where 0 = above threshold value and 1 = below threshold value
	"""
	# get band name from image
	if bandName == None:	
		bands = imageutils.get_band_names(inputImage)
		count = 0
		for name in bands:
			count += 1
			if count == imgBand:
				bN = name
	else:
		bN = bandName
	print(bN)
	# select chosen band to create output
	print('creating elevation mask...')
	imageutils.select_img_bands(inputImage, outputImage, gdalformat, rsgislib.TYPE_32FLOAT, [imgBand])
	imageutils.set_band_names(outputImage, [bN])
	# define band info for image mask
	bandDefns = []
	bandDefns.append(rsgislib.imagecalc.BandDefn(bandName, inputImage, imgBand))
	# define expression and classify image
	exp = '(' + bandName + '<' + str(eleThreshold) + ')?1:0'
	rsgislib.imagecalc.band_math(outputImage, exp, gdalformat, rsgislib.TYPE_8UINT, bandDefns)


def classify_otsu_multi(inputImage, imgBand, outputImage, gdalformat='KEA'):
    """
    Function that returns an classification image with 3 classes based on multi-level Otsu thresholding for use in remote sensing classification

    Args
    image - str, image containing band that otsu value will be generated from
    imgBand -  int, band number
    gdalformat - default is 'KEA'

    Returns
    outputImage - binary image where 0 = non classified pixels and 1 = classified pixels
    """

    # get band name from image
    bands = imageutils.get_band_names(inputImage)
    count = 0
    for name in bands:
        count += 1
        if count == imgBand:
            bandName = name
            print(name)
    # open image as raster surface 
    raster = gdal.Open(inputImage)
    # load image as array
    array = np.array(raster.GetRasterBand(imgBand).ReadAsArray())
    # remove nan and nodatavals
    array = array[~np.isnan(array)]
    array = array[array != -99]
    # define band info for image mask
    bandDefns = []
    bandDefns.append(rsgislib.imagecalc.BandDefn(bandName, inputImage, imgBand))
    # calculate multi otsu thresholds
    thresholds = filters.threshold_multiotsu(array)
    print(str(thresholds))
    thr1 = thresholds[0]
    thr2 = thresholds[1]
    print(str(thr1))
    print(str(thr2))
    # define expression and classify image
    exp = '(' + bandName + '== -99)?-99:(' + bandName + '<' + str(thr1) + ')?1: (' + bandName + '<' + str(thr2) + ')?2: (' + bandName + '>' + str(thr2) + ')?3:0'
    print(exp)
    print(outputImage)
    print(str(bandDefns))
    rsgislib.imagecalc.band_math(outputImage, exp, gdalformat, rsgislib.TYPE_8UINT, bandDefns)
    raster = None


def apply_otsu(inImg, outClass, outMskImg, band, maskVal=0, outVal=-99):
    """
    function that returns the class to be classified using otsu thresholding then masks input for next classification step

    Args 
    inImg - input composite image to be classified
    outClass - binary image where 0 = non classified pixels and 1 = classified pixels
    outMskImg - output image containing remaining pixels to be classified
    band - image band to use to calculate otsu threshold
    maskVal - is a float or list of floats representing the value(s) in the mask image that will be replaced with the outvalue.
    outVal - is a float representing the value written to the output image in place of the regions being masked.
 
    """
    # get band names from inImg
    bandNames = imageutils.get_band_names(inImg)
    # extract class by otsu and mask class
    classify_otsu(inImg, band, outClass)
    imageutils.mask_img(inImg, outClass, outMskImg, 'KEA', rsgislib.TYPE_32FLOAT, outVal, maskVal)
    imageutils.set_band_names(outMskImg, bandNames)
    #imageutils.pop_img_stats(outMskImg, True, outVal, True)
    set_nodata_val(outMskImg, outVal)


def apply_rules(inputImg, output_directory, water_indices_band=23, vegetation_indices_band=11, intertidal_indices_band=None, create_vegetation_class_img=False):
    """ 
    function that applies otsu thresholding rules to classify water, vegetation and intertidal if using a composite image, 
    returning binary classification images for each class and masked image containing remaining pixels to be classified via
    machine learning workflow.

    Args
        inputImg - str, filepath for input multispectral image to be classified 
        output_directory - str, specifying the directory where outputs will be saved
        input_img_prefix - boolean, if processing multiple images using multiprocessing function set to true and output directory will be created using input img prefix 
        water_indices_band - int, image band containing water indice (mndwi) used to classify water default=23 (mndwi_median) from annual Sentinel-2 composite image
        vegetation_indices_band - int, image band containing ndvi band used to classify vegetation default=11 (nvdi_intMn1090) from annual Sentinel-2 composite image
        intertidal_indices_band - int, image band containing water indice (ndwi_stddev) used to classify intertidal default=None if intertidal is not being classified.
        create_vegetation_class_img - boolean, if True outputs will contain composite image of vegetation pixels with all other pixels masked for further classification of vegetation. Default=false. 

    Returns
    binary image classifications for water, vegetation and intertidal areas

    """
    # make sure inputs use absolute paths as using working directory
    inputImg = os.path.abspath(inputImg)
    outDIR = os.path.abspath(output_directory)
    if not os.path.exists(outDIR):
        os.makedirs(outDIR)
    print(outDIR)
    

    # make sure inputs use absolute paths as using working directory
    # Create outDir from inputImg filename
    # classify water
    waterClass = os.path.join(outDIR, 'waterClass.kea')
    waterMaskImg = os.path.join(outDIR, 'waterMask.kea')
    apply_otsu(inputImg, waterClass, waterMaskImg, water_indices_band)

    # classify vegetation and generate supervised inputs
    vegClass = os.path.join(outDIR, 'vegClass.kea')
    mlInput = os.path.join(outDIR, 'supervisedInput.kea')
    apply_otsu(waterMaskImg, vegClass, mlInput, vegetation_indices_band)

    # create vegetation supervised input if specified
    if create_vegetation_class_img == True:
        vegmlInput = os.path.join(outDIR, 'vegetationSupervisedInput.kea')
        imageutils.maskImage(inputImg, vegClass, vegmlInput, 'KEA', rsgislib.TYPE_32FLOAT, -99, 1)
        # get band names from inputImage and apply to vegMLInput
        bandNames = imageutils.getBandNames(inputImg)
        imageutils.setBandNames(vegmlInput, bandNames)
        set_nodata_val(vegmlInput, -99) 

    # classify intertidal if using a composite image
    if not intertidal_indices_band == None:
        # mask inImg to get water area
        waterArea = os.path.join(outDIR, 'waterArea.kea')
        imageutils.mask_img(inputImg, waterClass, waterArea, 'KEA', rsgislib.TYPE_32FLOAT, -99, 1)
        # get & set band names from inputImg and set nodata
        bandNames = imageutils.get_band_names(inputImg)
        imageutils.set_band_names(waterArea, bandNames)
        set_nodata_val(waterArea, -99) 
        # multi-otsu threshold to separate water and tidal Classes
        waterTideClass = os.path.join(outDIR, 'waterTideClass.kea')
        classify_otsu_multi(waterArea, intertidal_indices_band, waterTideClass)
    
    # remove outputs that arent needed
    for i in waterArea, waterMaskImg:
        os.remove(i)

def apply_rules_mp(list_of_images, water_indices_band=23, vegetation_indices_band=11, intertidal_indices_band=None, num_of_processors=4):
    """
    Function to apply classification rules to a list of images returning a output directory with subdirectories containing all outputs for each image in list

    Args
    list_of_image - list of image strings
    water_indices_band - image band containing water indices that otsu will be applied to
    vegetation_indices_band -  image band containing vegetation indices that otsu will be applied to
    intertidal_indices_band - image band containing intertidal indices that otsu will be applied. When classifying a single image leave as default (None) 

    Returns
    output directory containing subdirectories with outputs for each image
    """
    # create and move into output directory
    outputs = 'outputs'
    if not os.path.exists(outputs):
        os.makedirs(outputs)
    os.chdir(outputs)

    # iterate over list_of_images and apply rules using partial and pool.map
    pool = multiprocessing.Pool(processes=num_of_processors)
    run_apply_rules = partial(apply_rules, outDir=outputs, water_indices_band=water_indices_band, vegetation_indices_band=vegetation_indices_band, intertidal_indices_band=intertidal_indices_band)
    pool.map(run_apply_rules, list_of_images)
    pool.close()
    pool.join()


def extractImgBandInfo(imgs):
    """
    A function that builds a list of imgBandInfo to extract training data from
    
    Args
    imgs = list of images to extract bandInfo from 
    
    Returns 
    list of rsgislib.imageutils.ImageBandInfo objects
    """
    # define return list
    imgInfo = []
    # iterate over each img in list
    for input in imgs:
        # Get a unique id for input
        uidStr = str(uuid.uuid4())
        # return list of band numbers
        numBand = len(imageutils.get_band_names(input))
        numBand += 1
        bands = range(1, numBand) 
        imgInfo.append(imageutils.ImageBandInfo(input, uidStr, bands))
    return imgInfo


def getTrainingData(inputImgList, vectorFile, vectorLayer, tmp=False, refImg=None, saveHDF5files=None):
    """
    Function that extracts training from vector shapefile for region defined by first image in inputImgList
    Vector must contain two attributes: className and classID for all classes to be identified

    Args
    inputImgList = list of images that training will be extracted from 
    vectorFile = shapefile containing training with two fields className and classID 
    vectorLayer - string containing name of the shapefile
    tmp = directory for temporary output. default is false
    refImg = reference image defining ROI, if none first image in image list will be used 
    saveHDF5files = filepath to folder if hdf5 are to be saved. If none files will be saved to tmp

    Returns
    dict containg rsgislib.classification.ClassSimpleInfoObj

    """
    if tmp == True:
        # Save current working path
        cPWD = os.getcwd()
        tmpDir = os.path.join(cPWD, 'tmp')
        # Create a working directory
        if not os.path.exists(tmpDir):
                    os.makedirs(tmpDir)
        # Move into that working directory.
        os.chdir(tmpDir)
    else:
        pass
    # generate valid mask if refImg not provided 
    if refImg == None:
        roi = 'roi.kea'
        rsgislib.imageutils.gen_valid_mask(inputImgList[0], roi, 'KEA', -99)
    else:
        roi = refImg
    # rasterise vectorFile
    vecRaster = 'vector.kea'
    if not os.path.exists(vecRaster):
        rsgislib.vectorutils.rasterise_vec_lyr(vectorFile, vectorLayer, roi, vecRaster, 'KEA', 0, vecAtt='classID', nodata=-99)
    # subset vecRaster to roi 
    trainingRaster = 'training.kea'
    if not os.path.exists(trainingRaster):
        rsgislib.imageutils.mask_img(vecRaster, roi, trainingRaster, 'KEA', rsgislib.TYPE_8UINT, -99, 0)
        rsgislib.rastergis.populate_stats(clumps=trainingRaster, addclrtab=True, calcpyramids=True, ignorezero=True)

    # extract training pixels
    # get imgInfo
    imgInfo = extractImgBandInfo(inputImgList)
    # create dict from shpfile containing className and classIDs
    openVector = gpd.read_file(vectorFile)
    classList = openVector['className'].unique()
    idList = openVector['classID'].unique()
    # Get pxlCount for each class in training raster
    count = rsgislib.imagecalc.count_pxls_of_val(trainingRaster, idList)
    countList = np.ndarray.tolist(count)
    # merge lists to dict with format className : [id, pxlCount]
    classDict = {classList[i]: [idList[i],countList[i]] for i in range(len(classList))}
    print(classDict)
    classTrainInfo = {}
    # create training extraction points and imgs for each class
    for name, idVal in classDict.items():
        print(idVal[0])
        print(type(idVal[0]))
        classRaster = name + '.kea'
        # create folder in outputs for hdf5 training files if specified
        fn = name + '.h5'
        if not saveHDF5files == None:
            h5Dir = os.path.join(saveHDF5files + '/training')
            classH5 = os.path.join(h5Dir, fn)
            if not os.path.exists(h5Dir):
                os.makedirs(h5Dir)
        else:
            classH5 = fn
        # training pxls = 25% of reference pxls
        numPxls = int(idVal[1]*0.25)
        print(numPxls)
        print(type(numPxls))
        if not os.path.exists(classH5):
            rsgislib.imageutils.perform_random_pxl_sample_in_mask_low_pxl_count(trainingRaster, classRaster, 'KEA', int(idVal[0]), numPxls, 50)
            rsgislib.rastergis.populate_stats(clumps=classRaster, addclrtab=True, calcpyramids=True, ignorezero=True)
            rsgislib.imageutils.extract_zone_img_band_values_to_hdf(imgInfo, classRaster, classH5, idVal[0])
        classTrainInfo[name] = rsgislib.classification.ClassSimpleInfoObj(id=idVal[0], fileH5=classH5, red=random.randint(0,255), green=random.randint(0,255), blue=random.randint(0,255))
    # remove tmp directory if it exists
    if os.path.exists(tmpDir):
        shutil.rmtree(tmpDir)
    
    return classTrainInfo


def genTrainingInfo(h5_folder):
    """
    Function that builds training info dict from h5 files for each class to train rsgislib.classification
    Inputs
    h5_folder - folder containing h5 files
    Returns
    dict - containing rsgislib.classification.ClassSimpleInfo objects to train classifier.
    """
    classTrainInfo = {}
    val = 0
    for h5_file in glob.glob(h5_folder + '/*.h5'):
        print(h5_file)
        name = h5_file.split('/')[-1].split('.')[0]
        print(name)
        out_id = val + 1
        classTrainInfo[name] = rsgislib.classification.ClassInfoObj(id=val, out_id=out_id, train_file_h5=h5_file, test_file_h5=h5_file, valid_file_h5=h5_file, red=random.randint(0,255), green=random.randint(0,255), blue=random.randint(0,255))
        val += 1
        print(val)

    return classTrainInfo


def classify_random_forest(input_image_list, input_training_images, training_folder=None):
    """
    Function to apply random forest classification to a list of images image using RSGISlib and scikit-learn gridsearch

    Args
    input_image_list - list of images containing the region to be classified this will be supervisedInput generated from the rules workflow 
    input_training_images - list of input training images to create bandinfo objects that matches training data bands
    training_folder - folder containing .h5 files to produce dict of training data. If None will use NZ training data folder

    Returns
    classification outputs
    """
    outDIR = 'outputs'
    # build training dict
    if training_folder == None:
        training_data = 'inputs/training-data/'
    else:
        training_data = training_folder
    
    print(training_data)
    
    training = genTrainingInfo(training_data)
    print(training)

    # get imgInfo for images
    imgInfo = extractImgBandInfo(input_training_images)
    print(imgInfo)

    # train classifier using grid search 
    gridSearch = GridSearchCV(RandomForestClassifier(), param_grid={'n_estimators':[10,20,50,100], 'max_depth':[2,4,8]})
    #classifier = RandomForestClassifier(n_estimators=100, max_depth=8)

    optimal_params = rsgislib.classification.classsklearn.perform_sklearn_classifier_param_search(training, gridSearch)
    rsgislib.classification.classsklearn.train_sklearn_classifier(training, optimal_params)

    # classify images in list
    for img in input_image_list:
        outName = os.path.join(outDIR, 'random_forest_output.kea') 
        # define mask of ROI
        roi = 'ROI.kea'
        # perform classification 
        print('classifying ' + img)
        imageutils.gen_valid_mask(img, roi, 'KEA', -99)
        rsgislib.classification.classsklearn.apply_sklearn_classifier(training, optimal_params, roi, 1, imgInfo, outName, gdalformat='KEA')  
        print('classified ' + img)
    # remove ROI
    os.remove(roi)

# define function to merge classes from folder
def merge(folder, output_format='KEA', exclude_intertidal_rule=False):
    """
    function to merge rule based and machine learning outputs into one classification output 
    
    Args
    folder - directory containing outputs to be merged
    output_format - output image format to use default = 'KEA'
    exclude_interidal_rule - if single time instance has been classified ignore intertidal rule default=False
    """

    # if exclude_intertidal rule==True define water cls image and expression accordingly
    if  exclude_intertidal_rule == True:
        water_cls = glob.glob(folder + '/*waterClass.kea')
        exp = """(water==0)?1:(veg==0)?2:(rmncls==1)?3:(rmncls==3)?4:(rmncls==2)?5:(urban==3)?6:
             (rmncls==6)?7:(rmncls==4)?8:(rmncls==5)?9:0"""
    else:
        water_cls = glob.glob(folder + '/*TideClass.kea')
        exp = """(water==1||water==2)?1:(veg==0)?2:(rmncls==1)?3:(rmncls==3)?4:(rmncls==2||water==3)?5:(urban==3)?6:
             (rmncls==6)?7:(rmncls==4)?8:(rmncls==5)?9:0"""
    
    # define other classification outputs
    veg_cls = glob.glob(folder + '/*vegClass.kea')
    other_cls = glob.glob(folder + '/*random_forest_output.kea')
    urban = glob.glob(folder + '/*urban.kea')
    
    print('veg', veg_cls)

    print('water', water_cls)

    # define bandDefns for bandmath
    bandDefns = []
    bandDefns.append(rsgislib.imagecalc.BandDefn('water', water_cls[0], 1))
    bandDefns.append(rsgislib.imagecalc.BandDefn('veg', veg_cls[0], 1))
    bandDefns.append(rsgislib.imagecalc.BandDefn('rmncls', other_cls[0], 1))
    bandDefns.append(rsgislib.imagecalc.BandDefn('urban', urban[0], 1))

    # bandmath to merge classes
    if output_format == 'KEA':
        file_name = 'classification_output.kea'
    if output_format == 'GTIFF':
        file_name = 'classification_output.tif'
    fnl_cls_img = os.path.join(folder, file_name)
    
    rsgislib.imagecalc.band_math(fnl_cls_img, exp, output_format, rsgislib.TYPE_8UINT, bandDefns)
    rsgislib.rastergis.pop_rat_img_stats(fnl_cls_img, add_clr_tab=True, calc_pyramids=True, ignore_zero=True)

    ratDataset = gdal.Open(fnl_cls_img, gdal.GA_Update)
    red = rat.readColumn(ratDataset, 'Red')
    green = rat.readColumn(ratDataset, 'Green')
    blue = rat.readColumn(ratDataset, 'Blue')
    ClassName = np.empty_like(red, dtype=np.dtype('a255'))
    ClassName[...] = ""


    red[1] = 51
    blue[1] = 153
    green[1] = 255
    ClassName[1] = 'Water'

    red[2] = 0
    blue[2] = 204
    green[2] = 0
    ClassName[2] = 'Vegetation'

    red[3] = 160
    blue[3] = 160
    green[3] = 160
    ClassName[3] = 'Dark sand'

    red[4] = 255
    blue[4] = 255
    green[4] = 51
    ClassName[4] = 'Light sand'

    red[5] = 0
    blue[5] = 0
    green[5] = 255
    ClassName[5] = 'Intertidal'

    red[6] = 255
    blue[6] = 255
    green[6] = 153
    ClassName[6] = 'Artificial surfaces'

    red[7] = 64
    blue[7] = 64
    green[7] = 64
    ClassName[7] = 'Bare rock'

    red[8] = 102
    blue[8] = 85
    green[8] = 0
    ClassName[8] = 'Gravel'

    red[9] = 85
    blue[9] = 128
    green[9] = 0
    ClassName[9] = 'Supratidal sand'

    rat.writeColumn(ratDataset, 'Red', red)
    rat.writeColumn(ratDataset, 'Green', green)
    rat.writeColumn(ratDataset, 'Blue', blue)
    rat.writeColumn(ratDataset, 'ClassName', ClassName)
    ratDataset = None