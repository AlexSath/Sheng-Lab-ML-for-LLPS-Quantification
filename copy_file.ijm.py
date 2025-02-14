import os
from ij import IJ
import shutil
from ij.io import FileSaver

#@ File (label = 'enter the input directory', style = 'directory') input_dir

def main():
	for root, dirs, files in os.walk(str(input_dir)):
		for f in files:
			if 'image.tif' in f:
				shutil.copyfile(os.path.join(root, f), os.path.join(root, 'merge.tif'))
				
main()