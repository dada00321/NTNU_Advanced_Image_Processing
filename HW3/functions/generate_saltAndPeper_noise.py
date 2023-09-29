import cv2
import numpy as np

# Generate salt-and-pepper noise
def salt_and_pepper(img, salt_prob=0.03, pepper_prob=0.03):
	h, w = img.shape[:2]
	total_pixels = int(h * w)

	noise = np.ones((h,w), dtype=np.uint8) * 127
	noisy_image = img.copy()

	# Add salt noise
	#num_salt = np.ceil(salt_prob * total_pixels)
	salt_coors = [np.random.randint(0, i-1, int(salt_prob * total_pixels)) for i in (h,w)]
	noisy_image[salt_coors[0], salt_coors[1], :] = 255
	noise[salt_coors[0], salt_coors[1]] = 255

	# Add pepper noise
	pepper_coors = [np.random.randint(0, i-1, int(pepper_prob * total_pixels)) for i in (h,w)]
	noisy_image[pepper_coors[0], pepper_coors[1], :] = 0
	noise[pepper_coors[0], pepper_coors[1]] = 0

	noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
	return noisy_image, noise

img_path = "C:/Users/USER/Pictures/woman-face-2.jpg"
input_image = cv2.imread(img_path)

# Add salt-and-pepper noise to the image
salt_prob = 0.03  # Adjust this value to control the salt noise level
pepper_prob = 0.03  # Adjust this value to control the pepper noise level
noisy_image, noise = salt_and_pepper(input_image, salt_prob, pepper_prob)

# Step 3: Save or display the noisy image
#cv2.imwrite("noisy_image.jpg", noisy_image)
cv2.imshow("Noisy Image", noisy_image)
cv2.imshow("Noise", noise)
cv2.waitKey(0)
cv2.destroyAllWindows()
