from os import listdir
from os.path import join, isdir
from shutil import copy

#@ File (label="Please enter directory with 'image.tif'", style="directory") image_dir
#@ File (label="Please enter directory with classes", style="directory") pred_dir

image_dir = str(image_dir)
pred_dir = str(pred_dir)

def main():
	for img_dir in listdir(image_dir):
		if img_dir == '.DS_Store':
			continue
		if not isdir(join(pred_dir, img_dir)):
			raise ValueError("Directory: '" + img_dir + "' does not exist at target")
		copy(join(image_dir, img_dir, 'image.tif'), join(pred_dir, img_dir))
		
main()
