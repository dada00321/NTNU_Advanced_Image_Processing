import numpy as np
import cv2

def convolve(img, kernel):
	# Create an empty output image
	convolved_img = np.zeros_like(img)

	# Flip the kernel horizontally and vertically (convolution theorem)
	kernel = np.flipud(np.fliplr(kernel))

	# Define padding size (e.g., 1 for a 3x3 kernel)
	padding = len(kernel) // 2

	# Apply zero-padding to the image
	padded_img = np.pad(img, ((padding, padding), (padding, padding)), mode="constant")

	# Get dimensions of the image and kernel
	img_h, img_w = padded_img.shape
	kernel_h, kernel_w = kernel.shape

	"""
	Perform convolution using nested loops
	"""
	for y in range(0, img_h - kernel_h + 1):
	    for x in range(0, img_w - kernel_w + 1):
	        # Extract the region of interest (ROI) from the padded image
	        roi = padded_img[y : (y + kernel_h), x : (x + kernel_w)]

	        # Perform element-wise multiplication and sum to get the result
	        convolved_img[y, x] = np.sum(roi * kernel)

	# PS: Alternative of the operation above
	# output_image = cv2.filter2D(image, -1, kernel)

	'''
	cv2.imshow("Original image", img)
	#cv2.imshow("Padded image", padded_img)
	cv2.imshow("Convolved image", convolved_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''
	return convolved_img

if __name__ == "__main__":
	# Read an image
	'''
	img_path = "C:/Users/USER/Pictures/umi.png"
	img_path = "C:/Users/USER/Pictures/a.jpg"
	img_path = "C:/Users/USER/Pictures/chess.jpg"
	img_path = "C:/Users/USER/Pictures/taipei-station.jpg"
	img_path = "C:/Users/USER/Pictures/lib.jpg"
	img_path = "C:/Users/USER/Pictures/lib-2.jpg"
	'''
	img_path = "C:/Users/USER/Pictures/bricks.jpg"



	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	
	img = np.array([[1,2,3],
				    [4,5,6],
				    [7,8,9]],dtype="uint8")

	''' Define a kernel (3x3 Gaussian blur kernel as an example) '''
	"""
	kernel = np.array([[1, 2, 1],
	                   [2, 4, 2],
	                   [1, 2, 1]])
	"""
	
	"""
	kernel = np.array([[-2,-1, 0, 1, 2],
	                   [-2,-1, 0, 1, 2],
					   [-2,-1, 0, 1, 2],
					   [-2,-1, 0, 1, 2],
	                   [-2,-1, 0, 1, 2]])
	"""

	#kernel = np.ones((5,5),dtype=np.uint8) / 25

	#kernel = np.ones((7,7),dtype=np.uint8) / 49
	'''
	kernel = np.array([[1/25, 1/25, 1/25, 1/25, 1/25],
	                   [1/25, 1/25, 1/25, 1/25, 1/25],
					   [1/25, 1/25, 1/25, 1/25, 1/25],
					   [1/25, 1/25, 1/25, 1/25, 1/25],
					   [1/25, 1/25, 1/25, 1/25, 1/25]])


	'''

	'''
	kernel = 3*np.array([[-0.25, -0.2,   0.,    0.2,   0.25],
				       [-0.4,  -0.5,   0.,    0.5,   0.4 ],
				       [-0.5,  -1.,    0.,    1.,    0.5 ],
				       [-0.4,  -0.5,   0.,    0.5,   0.4 ],
				       [-0.25, -0.2,   0.,    0.2,   0.25]])

	kernel = 3*np.array([[-0.25, -0.4,   -0.5,  -0.4,  -0.25],
				       [-0.2,  -0.5,  -1,   -0.5,  -0.2],
				       [0,    0,    0,    0,    0],
				       [0.2,   0.5,   1,    0.5,   0.2],
				       [0.25,  0.4,   0.5,   0.4,   0.25]])
	'''

	'''
	kernel = np.array([[2,2,4,2,2],
				       [1,1,2,1,1],
				       [0,0,0,0,0],
				       [-1,-1,-2,-1,-1],
				       [-2,-2,-4,-2,-2]])

	kernel = np.array([[2,1,0,-1,-2],
				       [2,1,0,-1,-2],
				       [4,2,0,-2,-4],
				       [2,1,0,-1,-2],
				       [2,1,0,-1,-2]])
	'''

	"""
	kernel = np.array([[5,8,10,8,5],
				       [4,10,20,10,4],
				       [0,0,0,0,0],
				       [-4, -10, -20, -10, -4],
				       [-5, -8, -10, -8, -5]])
	"""


	# Sobel filter (G_y)
	"""
	# https://www.researchgate.net/publication/49619233_Image_Segmentation_using_Extended_Edge_Operator_for_Mammographic_Images/figures?lo=1
	kernel = np.array([[2, 1, 0, -1, -2],
			           [2, 1, 0, -1, -2],
			           [4, 2, 0, -2, -4],
			           [2, 1, 0, -1, -2],
			           [2, 1, 0, -1, -2]])

	# Laplacian filter
	# https://kalpanileo1996.medium.com/edge-detection-with-laplacian-operator-without-using-opencv-laplacian-inbuilt-function-fa6d4966065f
	kernel = np.array([[0, 0, -1, 0, 0],
				       [0, -1, -2 ,-1, 0],
				       [-1, -2, 16, -2, -1],
				       [0, -1, -2, -1, 0],
				       [0, 0, -1, 0, 0]])
	"""
	
	#kernel = np.ones((13,13),dtype=np.uint8) / 169
	"""
	kernel = np.array([[0, -1, 0],
					   [-1, 5, -1],
					   [0, -1, 0]])
	"""
	
	kernel = np.array([[-1,-2,-1],
					   [0,0,0],
					   [1,2,1]])

	''' Convolution '''
	convolved_img = convolve(img, kernel)
	print(convolved_img)