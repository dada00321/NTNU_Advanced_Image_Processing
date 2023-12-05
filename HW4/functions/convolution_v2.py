import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def plot_image(img: np.array):
	plt.figure(figsize=(6, 6))
	plt.imshow(img, cmap="gray");

def plot_two_images(img1: np.array, img2: np.array):
	_, ax = plt.subplots(1, 2, figsize=(12, 6))
	ax[0].imshow(img1, cmap="gray")
	ax[1].imshow(img2, cmap="gray")

def calculate_target_size(img_size: int, kernel_size: int) -> int:
	num_pixels = 0

	# From 0 up to img size (if img size = 224, then up to 223)
	for i in range(img_size):
		# Add the kernel size (let"s say 3) to the current i
		added = i + kernel_size
		# It must be lower than the image size
		if added <= img_size:
			# Increment if so
			num_pixels += 1

	return num_pixels

def convolve(img: np.array, kernel: np.array) -> np.array:
	# Assuming a rectangular image
	tgt_size = calculate_target_size(
		img_size=img.shape[0],
		kernel_size=kernel.shape[0]
	)
	# To simplify things
	k = kernel.shape[0]

	# 2D array of zeros
	convolved_img = np.zeros(shape=(tgt_size, tgt_size))

	# Iterate over the rows
	for i in range(tgt_size):
		# Iterate over the columns
		for j in range(tgt_size):
			# img[i, j] = individual pixel value
			# Get the current matrix
			mat = img[i:i+k, j:j+k]

			# Apply the convolution - element-wise multiplication and summation of the result
			# Store the result to i-th row and j-th column of our convolved_img array
			convolved_img[i, j] = np.sum(np.multiply(mat, kernel))

	return convolved_img

def get_padding_width_per_side(kernel_size: int) -> int:
    # Simple integer division
    return kernel_size // 2

def add_padding_to_image(img: np.array, padding_width: int) -> np.array:
    # Array of zeros of shape (img + padding_width)
    img_with_padding = np.zeros(shape=(
        img.shape[0] + padding_width * 2,  # Multiply with two because we need padding on all sides
        img.shape[1] + padding_width * 2
    ))

    # Change the inner elements
    # For example, if img.shape = (224, 224), and img_with_padding.shape = (226, 226)
    # keep the pixel wide padding on all sides, but change the other values to be the same as img
    img_with_padding[padding_width:-padding_width, padding_width:-padding_width] = img

    return img_with_padding

if __name__ == "__main__":
	sharpen = np.array([
		[0, -1, 0],
		[-1, 5, -1],
		[0, -1, 0]
	])

	blur = np.array([
		[0.0625, 0.125, 0.0625],
		[0.125,  0.25,  0.125],
		[0.0625, 0.125, 0.0625]
	])

	outline = np.array([
		[-1, -1, -1],
		[-1,  8, -1],
		[-1, -1, -1]
	])
	
	##################################
	w1 = np.array([[-1,-2,-1],
		           [0,0,0],
			       [1,2,1]])
	
	w2 = np.array([[1,1,3],
		           [4,5,5],
			       [3,2,2]])
	test_kernel = w2
	
	# Flip the matrix vertically
	flipped_vertical = np.flip(test_kernel, axis=0)
	
	# Flip the vertically flipped matrix horizontally
	test_kernel = flipped_twice = np.flip(flipped_vertical, axis=1)
	
	'''
	img = Image.open("cat-1.jpg")
	img = ImageOps.grayscale(img)
	img = img.resize(size=(224, 224))
	'''
	
	##
	I1 = np.array([[1,2,3],
				   [4,5,6],
				   [7,8,9]],dtype="uint8")
	KERNEL_SIZE = 3
	
	##	
	I2 = np.array([[0,0,0,0,0,0],
				   [0,0,0,0,0,0],
				   [0,1,2,0,0,0],
				   [0,0,0,0,0,0],
				   [0,0,0,2,0,0],
				   [0,0,0,0,0,0]],dtype="uint8")
	KERNEL_SIZE = 6
	
	##
	img = I2
	
	##
	pad_img = get_padding_width_per_side(kernel_size=KERNEL_SIZE)
	img_with_padding= add_padding_to_image(
	    img=np.array(img),
	    padding_width=pad_img
	)
	plot_image(img=img)
	dst_img = convolve(img=img_with_padding, kernel=test_kernel)
	print(dst_img)
	
	'''
	plot_two_images(img1=img, img2=img_sharpened)

	pad_3x3 = get_padding_width_per_side(kernel_size=3)
	img_with_padding_3x3 = add_padding_to_image(
	    img=np.array(img),
	    padding_width=pad_3x3
	)
	print(img_with_padding_3x3.shape)
	plot_image(img_with_padding_3x3)

	pad_5x5 = get_padding_width_per_side(kernel_size=5)
	img_with_padding_5x5 = add_padding_to_image(
	    img=np.array(img),
	    padding_width=pad_5x5
	)
	print(img_with_padding_5x5.shape)
	plot_image(img_with_padding_5x5)

	img_padded_3x3_sharpened = convolve(img=img_with_padding_3x3, kernel=sharpen)
	print(img_padded_3x3_sharpened.shape)

	plot_two_images(
	    img1=img,
	    img2=img_padded_3x3_sharpened
	)
	'''
	#plot_image([[1,2,3],[4,5,6],[7,8,9]])
