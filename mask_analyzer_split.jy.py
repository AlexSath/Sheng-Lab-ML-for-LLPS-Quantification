### JAVA IMPORTS
from java.awt.Color import BLACK, GREEN, MAGENTA, YELLOW
from java.util import Arrays

### IMAGEJ IMPORTS
from ij import IJ, ImagePlus, WindowManager
from ij.gui import Roi, ShapeRoi, PolygonRoi, WaitForUserDialog, ShapeRoi, WaitForUserDialog
from ij.plugin.filter import Analyzer, ParticleAnalyzer, Filler
from ij.plugin.frame import RoiManager
from ij.measure import ResultsTable
from ij.measure.Measurements import *
from ij.plugin import Thresholder, RGBStackMerge, RGBStackConverter
from ij.macro import Interpreter 
from ij.process import ImageProcessor
import os

### ON STARTUP ###
# Go into batch mode
Interpreter.batchMode = True

# Clear all windows
while type(None) != type(WindowManager.getCurrentImage()):
	img = WindowManager.getCurrentImage()
	img.close()
	
if WindowManager.getWindow("Results"):
	WindowManager.getWindow("Results").close()

# Set up empty ROI Manager (instance stored in global 'RM')
if type(None) == type(RoiManager.getInstance()):
	RM = RoiManager()
else:
	RM = RoiManager.getInstance()
	RM.reset()

### USER INPUTS ###
#@ File (label="Please enter input directory", style="directory") input_dir
#@ File (label="Please enter output directory", style="directory") output_dir
#@ Boolean (label="Save pseudo-colored mask?") ps_bool

### MAIN FUNCTION ###
def main():
	# images in order are base, punctate mask, diffuse mask, and image with cell structure
	bases = ['image.tif', 'image_c0.tif', 'image_c1.tif', 'merge.tif']
	
	# find the paths for all image directories with the necessary basenames
	imgdirlist, excluded_dirs = get_dirs(input_dir, bases)	
				
	bound = []
	
	# Create aggregate roi data tables (one roi per row)
	pRes = ResultsTable()
	dRes = ResultsTable()
	
	# Create summary tables (one image per row)
	pSum = ResultsTable()
	dSum = ResultsTable()
	
	# main loop iterates through images in their respective directories
	for idx, imgdir in enumerate(imgdirlist):
		imgName = os.path.basename(imgdir)
	
		#Create nucleus and soma ROIs
		Interpreter.batchMode = False
		bound = select_soma(os.path.join(imgdir, bases[3]))
		
		# if pseudo-colored boolean should be created, do it here
		Interpreter.batchMode = True
		if ps_bool:
			pseudo_color(os.path.join(imgdir, "pseudo-color.tif"), bound,
						 os.path.join(imgdir, bases[1]), os.path.join(imgdir, bases[2]))
						 
		# measure punctate and diffuse rois
		pResTemp, pSumTemp, pRois = measure_mask(pRes, pSum, os.path.join(imgdir, bases[0]), os.path.join(imgdir, bases[1]), bound)
		aggRes = copy_table(pResTemp, pRes, imgName)
		aggSum = copy_table(pSumTemp, pSum, imgName)
		
		dResTemp, dSumTemp, dRois = measure_mask(dRes, dSum, os.path.join(imgdir, bases[0]), os.path.join(imgdir, bases[2]), bound)
		aggRes = copy_table(dResTemp, dRes, imgName)
		aggSum = copy_table(dSumTemp, dSum, imgName)
		
		merge = IJ.openImage(os.path.join(imgdir, bases[3]))
		IJ.run(merge, "RGB Color", "")
		mergeProc = merge.getProcessor()
		
		mergeProc.setColor(YELLOW)
		for roi in bound:
			roi.drawPixels(mergeProc)
		
		mergeProc.setColor(GREEN)
		for roi in pRois:
			roi.drawPixels(mergeProc)
			
		mergeProc.setColor(MAGENTA)
		for roi in dRois:
			roi.drawPixels(mergeProc)
		
		IJ.save(merge, os.path.join(imgdir, "merge_rois.tif"))
	
	pRes.save(os.path.join(str(output_dir), "punctate_res.csv"))
	dRes.save(os.path.join(str(output_dir), "diffuse_res.csv"))
	pSum.save(os.path.join(str(output_dir), "punctate_sum.csv"))
	dSum.save(os.path.join(str(output_dir), "diffuse_sum.csv"))
	
	Interpreter.batchMode = False



### FUNCTION GET_DIRS 
# Description:
# pre-conditions: takes list of tif basenames
# post-conditions:
def get_dirs(input_dir, bases):
	imgdirlist = []
	excluded_dirs = []
	for root, dirs, files in os.walk(str(input_dir)):
			for d in dirs:
				for base in bases:
					if not os.path.isfile(os.path.join(root, d, base)):
						excluded_dirs.append(os.path.join(root, d))
						break
				else:	
					imgdirlist.append(os.path.join(root, d))
	return imgdirlist, excluded_dirs


### FUNCTION PSEUDO_COLOR
# Description:
# pre-conditions:
# post-conditions:
def pseudo_color(savepath, boundaries, *masks):
	COLORS = ['red', 'green', 'blue', 'gray', 'cyan', 'magenta', 'yellow']
	colors = []
	if len(masks) == 2:
		colors = ['green', 'magenta']
	else:
		raise(ValueError, len(masks) + " masks is a number without hardcoded pseudo-colors")
	
	#TODO: Ensure appended image plus are the same size as the images of interest
	temp = IJ.openImage(masks[0])
	temp.getProcessor().set(0)
	
	images = []
	for C in COLORS:
		if C in colors:
			img = IJ.openImage(masks[colors.index(C)])
			imgProc = img.getProcessor()
			imgProc.setBackgroundColor(BLACK)
			imgProc.fillOutside(boundaries[0])
			imgProc.fill(boundaries[1])
			imgProc.scaleAndSetThreshold(124, 255, ImageProcessor.RED_LUT)
			maskProc = imgProc.createMask()
			maskProc.setBackgroundColor(BLACK)
			img.setProcessor(maskProc)
			images.append(img)
		else:
			images.append(ImagePlus("empty", temp.getProcessor()))
			
	rgbPlus = RGBStackMerge().mergeChannels(images, False)
	RGBStackConverter().convertToRGB(rgbPlus)
	IJ.save(rgbPlus, savepath)


### FUNCTION SELECT_SOMA
# Description:
# pre-conditions:
# post-conditions:
def select_soma(mergepath):
	IJ.setTool("polygon")
	merge = IJ.openImage(mergepath)
	merge.show()
	
	user_dialog = WaitForUserDialog("Please select the area for the soma")
	user_dialog.show()
	roi_outer = merge.getRoi()

	user_dialog = WaitForUserDialog("Please select the area for the nucleus")
	user_dialog.show()
	roi_inner = merge.getRoi()

	merge.close()
	return [roi_outer, roi_inner]


### FUNCTION MEASURE_MASK
# Description:
# pre-conditions:
# post-conditions:
def measure_mask(aggRes, aggSum, imagepath, maskpath, boundaries):
	Interpreter.batchMode = True
	RM.reset()
	
	dirname = os.path.basename(os.path.dirname(imagepath))
	image = IJ.openImage(imagepath)
	mask = IJ.openImage(maskpath)
	mask.setCalibration(image.getCalibration())
	
	outer, inner = get_soma_measurements(mask, boundaries)
	
	if image.width != mask.width or image.height != mask.height:
		image = image.resize(mask.width, mask.height, "bilinear")
		image.setTitle(os.path.basename(imagepath))
	
	maskProc = mask.getProcessor()
	maskProc.setBackgroundColor(BLACK)
	maskProc.fillOutside(boundaries[0])
	maskProc.fill(boundaries[1])
	
	# NOTE: Thresholding assumes that mask comes from machine learning prediction
	# where pixel intensity is proportional to the certainty of the model based on
	# several predictions with range of input patches.
	maskProc.scaleAndSetThreshold(124, 255, ImageProcessor.RED_LUT)

	# TODO: Ensure that analyzer has correct image scale
	rT, sT = ResultsTable(), ResultsTable()
	options = ParticleAnalyzer.ADD_TO_MANAGER + ParticleAnalyzer.DISPLAY_SUMMARY
	measurements = AREA + CIRCULARITY + INTEGRATED_DENSITY + MEAN + MEDIAN + MIN_MAX + SHAPE_DESCRIPTORS
	# minimum diameter of 0.002 is ~= surface area of particle with diameter of 50nm
	pa = ParticleAnalyzer(options, measurements, rT, 0.00196, 10000000, 0, 1)
	Analyzer.setRedirectImage(image)
	pa.setSummaryTable(sT)
	pa.analyze(mask)
	
	# add nuclear area and soma area
	tableCount = sT.size()
	sT.setValue("nuclearArea", tableCount - 1, inner)
	sT.setValue("somaArea", tableCount - 1, outer - inner)
	
	image.close()
	mask.close()

	return rT, sT, RM.getRoisAsArray()


### FUNCTION GET_SOMA_MEASUREMENTS
# Description:
# pre-conditions:
# post-conditions:
def get_soma_measurements(img, rois):
	for roi in rois:
		RM.addRoi(roi)
	IJ.run("Set Measurements...", "area mean shape integrated limit decimal=3")
	resTable = RM.multiMeasure(img)
	outer = resTable.getValue("Area1", 0)
	inner = resTable.getValue("Area2", 0)
	RM.reset()
	return outer, inner


### FUNCTION COPY_TABLE
# Description:
# pre-conditions:
# post-conditions:
def copy_table(sourceTable, targetTable, sourceName = ""):
	targetCount = targetTable.getCounter()
	cols = sourceTable.getHeadings()
	for i in range(sourceTable.getCounter()):
		if sourceName != "":
			targetTable.setValue("Source", targetCount + i, sourceName)
		for col in cols:
			result = sourceTable.getValue(col, i)
			targetTable.setValue(col, targetCount + i, result)
	
	return targetTable

### FUNCTION SHOW_IMG
# Description:
# pre-conditions:
# post-conditions:
def show_img(img):
	Interpreter.batchMode = False
	img.show()
	user_dialog = WaitForUserDialog("")
	user_dialog.show()
	img.close()
	Interpreter.batchMode = True


# Start main function
main()