import cv2
import numpy as np
import math

def generate_gaussian_noise(img, G=200, sigma=50):
	noise = img.copy()
	noisy_img = img.copy()

	noisy_img = noisy_img.astype("float32")

	noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY)

	h, w = noisy_img.shape[:2]
	num_of_pixels = h * w

	num_of_counts = num_of_pixels // 2
	if num_of_pixels % 2 != 0:
		num_of_counts += 1

	count = 0
	for i in range(h):
		for j in range(w):
			fi, r = np.random.ranf(), np.random.ranf()

			#print(f"[{i},{j}]")
			if count % 2 == 0:
				z1 = sigma * math.cos(2 * math.pi * fi) * (-2 * math.log(r))**0.5
				tmp = noisy_img[i,j] + z1
				noise[i,j] = z1
				count += 1

			else:
				z2 = sigma * math.sin(2 * math.pi * fi) * (-2 * math.log(r))**0.5
				tmp = noisy_img[i,j] + z2
				noise[i,j] = z2

			if tmp < 0: tmp = 0
			elif tmp > (G-1): tmp = (G-1)

			noisy_img[i,j] = tmp

	noisy_img = np.clip(noisy_img, 0, 255).astype("uint8")
	return noisy_img, noise

if __name__ == "__main__":
	img_path = "C:/Users/USER/Pictures/woman-face-2.jpg"
	#img_path = "C:/Users/USER/Pictures/umi.png"

	raw_img = cv2.imread(img_path)

	noisy_img, noise = generate_gaussian_noise(raw_img)


	cv2.imshow("raw image", raw_img)
	cv2.imshow("noise", noise)
	cv2.imshow("noisy image", noisy_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''
	'''


