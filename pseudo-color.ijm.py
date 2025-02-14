import os
from ij import IJ
from ij import WindowManager
from ij import ImagePlus

#@ File (style = 'directory') input_dir

COLORS = ['red', 'green', 'blue', 'gray', 'cyan', 'magenta', 'yellow']
imgnames = ['image_c0.tif', 'image_c1.tif']

def main():
	for direc in os.listdir(str(input_dir)):
		if not os.path.isdir(os.path.join(str(input_dir), direc)):
			continue
			
		dirpath = os.path.join(str(input_dir), direc)
		rgb = colorize(dirpath, ("green", imgnames[0]), ('magenta', imgnames[1]))
		IJ.save(rgb, os.path.join(dirpath, 'pseudo-color.tif'))
		rgb.close()
		
def colorize(direc, *colors):
	string = ""
	for c in colors:
		idx = COLORS.index(c[0])
		string += "c" + str(idx + 1) + "=" + c[1] + " "
		this_img = IJ.openImage(os.path.join(direc, c[1]))
		this_img.show()
	string += "create"
	IJ.run("Merge Channels...", string)
	img = WindowManager.getImage("Composite")
	IJ.run(img, "RGB Color", "")
	rgbPlus = WindowManager.getImage("Composite (RGB)")
	img.close()
	return rgbPlus
	
main()