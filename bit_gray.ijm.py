import os
from ij import IJ
from ij.io import FileSaver

#@ File (label = 'enter the input directory', style = 'directory') input_dir

def main():
	for root, dirs, files in os.walk(str(input_dir)):
		for f in files:
			if 'image.tif' in f:
				img = IJ.openImage(os.path.join(root, f))
				IJ.run(img, "8-bit", "")
				IJ.run(img, "Grays", "")
				f = FileSaver(img)
				f.save()
				
main()