import os
from ij import IJ

#@ File (label = 'enter the input directory', style = 'directory') input_dir
#@ File (label = 'enter save directory', style = 'directory') output_dir
#@ String (label = 'enter image save name') save_name

def main():
	for root, dirs, files in os.walk(str(input_dir)):
		for f in files:
			if '.tif' in f:
				base = root[len(input_dir):]
				base.replace(os.path.sep, '_')
				this_out_path = os.path.join(output_dir, base, savename)
				
				img = IJ.openImage(os.path.join(root, f))
				IJ.save(img, this_out_path)
				
main()