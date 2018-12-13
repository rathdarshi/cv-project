image_paths = glob("./runs/1544681386.8921719/")
label_paths = {
	re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
	for path in glob(os.path.join("../data_road/training/", 'gt_image_2', '*_road_*.png'))}
background_color = np.array([255, 0, 0])
# Loop through batches and grab images, yielding each batch
for i in range(0, len(image_paths)):
	#images = []
	#gt_images = []
	image_file = image_paths[i]
	gt_image_file = label_paths[os.path.basename(image_paths[i])]
	
	# Re-size to image_shape
	image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
	gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

	# Create "one-hot-like" labels by class
	gt_bg = np.all(gt_image == background_color, axis=2)
	gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
	gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)



	image = np.array(image)
	gt_image = np.array(gt_image)


	print(image)
	print(gt_image)


