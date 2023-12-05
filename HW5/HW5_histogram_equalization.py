"""
Advanced Image Processing - HW4

> Title of the main window: `AIP-NTUST_M11125014`
> File name of the .exe file: `HW1-NTUST_M11125014.exe`
> Input image extensions (at least): JPG, BMP, PPM
> Output: (No special restrictions)
> Function-1: Read image
> Function-2: Rotate image
> Function-3: Identify the image format & image size
> Function-4: Draw histogram(s)
> Function-5: Generate noise
> Function-6: Convolution & Detect Edges
"""

import sys
import os
import shutil

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore  import *

import imutils
import cv2
#from skimage.io import imread
#from skimage import img_as_ubyte
#import skimage

from matplotlib import pyplot as plt
import numpy as np
import math

class ConvolveWindow(QWidget):
	confirm_signal = pyqtSignal()  # Define a signal

	def __init__(self):
		super().__init__()
		self.kernel = None

		self.setWindowTitle("Convolution")
		self.setGeometry(100, 100, 300, 415)

		self.layout = QVBoxLayout()

		self.btnKernel_5x5 = QRadioButton("5x5")
		self.btnKernel_5x5.clicked.connect(self.enable_kernel)
		self.btnKernel_7x7 = QRadioButton("7x7")
		self.btnKernel_7x7.clicked.connect(self.enable_kernel)
		self.layout.addWidget(self.btnKernel_5x5, 0, Qt.AlignLeft)
		self.layout.addWidget(self.btnKernel_7x7, 0, Qt.AlignLeft)

		# =============================================================================
		num = 7

		# Create a table
		self.table = QTableWidget(num, num)
		self.table.setMaximumHeight(290)
		self.table.setMaximumWidth(290)
		#self.table.setVisible(False)

		column_width = 33  # Adjust this value as needed
		for col in range(num):
			self.table.setColumnWidth(col, column_width)

		for row in range(num):
			for col in range(num):
				item = QTableWidgetItem("X")
				item.setForeground(Qt.red)
				item.setFlags(item.flags() & ~Qt.ItemIsEditable) # not editable
				item.setTextAlignment(Qt.AlignCenter)  # Align center vertically and horizontally
				self.table.setItem(row, col, item)

		self.layout.addWidget(self.table, 0, Qt.AlignLeft)
		# =============================================================================

		#layout.addWidget(self.label)
		#layout.addWidget(self.text_edit)
		#layout.addWidget(sub_layout_1)
		self.button = QPushButton("確定")
		self.button.clicked.connect(self.hideWindow)
		self.layout.addWidget(self.button, 100, Qt.AlignLeft)

		self.setLayout(self.layout)

	def enable_kernel(self):
		option_1 = self.btnKernel_5x5.isChecked()
		option_2 = self.btnKernel_7x7.isChecked()

		if any([option_1, option_2]):
			if option_1:
				num = 5
				self.table.setHorizontalHeaderLabels([str(i) for i in range(1, 6)]+['', ''])
				self.table.setVerticalHeaderLabels([str(i) for i in range(1, 6)]+['', ''])

			else:
				num = 7
				self.table.setHorizontalHeaderLabels([str(i) for i in range(1, 8)])
				self.table.setVerticalHeaderLabels([str(i) for i in range(1, 8)])

			self.kernel_num = num

			for row in range(7):
				for col in range(7):
					if row >= num or col >= num:
						item = QTableWidgetItem("X")
						item.setForeground(Qt.red)
						item.setFlags(item.flags() & ~Qt.ItemIsEditable) # not editable
					else:
						item = QTableWidgetItem()
						item.setFlags(item.flags() | 0x0002) # editable
					item.setTextAlignment(Qt.AlignCenter)  # Align center vertically and horizontally
					self.table.setItem(row, col, item)

			#print(num)
			#self.table.setVisible(True)

	def get_kernel(self):
		kernel = []
		is_stopped = False
		for row in range(self.kernel_num):
			row_numbers = []
			for col in range(self.kernel_num):
				item = self.table.item(row, col)
				if item is not None:
					try:
						raw_text = item.text()

						if '/' not in raw_text:
							number = float(raw_text)
						else:
							a, b = map(int, raw_text.split('/'))
							number = float(a) / b

						row_numbers.append(number)
					except ValueError:
						is_stopped = True

						# Handle invalid input if needed
						QMessageBox.information(self, "錯誤訊息", "輸入格式有誤，請確認輸入皆必須為數字", QMessageBox.Ok)
			kernel.append(row_numbers)
		if not is_stopped:
			self.kernel = kernel

	def hideWindow(self):
		self.get_kernel()

		#print(f"kernel:\n{self.kernel}")

		self.confirm_signal.emit()
		#print(self.btnGaussainNoise.isChecked())
		#print(self.btnPepperSaltNoise.isChecked())
		self.hide()

class NoiseGenerateWindow(QWidget):
	confirm_signal = pyqtSignal()  # Define a signal

	def __init__(self):
		super().__init__()

		self.setWindowTitle("Gaussian Noise")
		self.setGeometry(100, 100, 200, 100)

		layout = QVBoxLayout()

		self.btnGaussainNoise = QRadioButton("高斯雜訊")
		self.btnGaussainNoise.clicked.connect(self.enableStandardDeviationTextEdit)
		self.btnPepperSaltNoise = QRadioButton("椒鹽雜訊")
		self.btnPepperSaltNoise.clicked.connect(self.disableStandardDeviationTextEdit)
		layout.addWidget(self.btnGaussainNoise, 0, Qt.AlignLeft)
		layout.addWidget(self.btnPepperSaltNoise, 0, Qt.AlignLeft)

		w1 = QWidget()
		self.standard_deviation = QLineEdit()
		self.standard_deviation.setFixedWidth(60)
		self.standard_deviation.setReadOnly(True)
		sub_layout_1 = QFormLayout()
		sub_layout_1.addRow("標準差 (σ):", self.standard_deviation)
		w1.setLayout(sub_layout_1)
		layout.addWidget(w1, 0, Qt.AlignLeft)

		#layout.addWidget(self.label)
		#layout.addWidget(self.text_edit)
		#layout.addWidget(sub_layout_1)
		self.button = QPushButton("確定")
		self.button.clicked.connect(self.hideWindow)
		layout.addWidget(self.button, 100, Qt.AlignLeft)

		self.setLayout(layout)

	def enableStandardDeviationTextEdit(self):
		self.standard_deviation.setReadOnly(False)

	def disableStandardDeviationTextEdit(self):
		self.standard_deviation.setReadOnly(True)

	def hideWindow(self):
		self.confirm_signal.emit()
		#print(self.btnGaussainNoise.isChecked())
		#print(self.btnPepperSaltNoise.isChecked())
		self.hide()

class ImageProcess():
	def set_image(self, img_path):
		self.img = cv2.imread(img_path)

	def rotate(self, rotation_angle):
		# Rotate the image
		rotated_img = imutils.rotate_bound(self.img, rotation_angle)
		return rotated_img
	
	# Draw histogram of image with Histogram Eequalization
	def draw_and_save_grayHist_with_equalization(self, save_dir, img_data=None, img_path=""):
		if img_data is None:
			equalized_grayHist_path = f"{save_dir}/equalized_gray_level_histogram.png"
			if os.path.exists(equalized_grayHist_path):
				os.remove(equalized_grayHist_path)
			target_img = self.img
		else:
			equalized_grayHist_path = f"{save_dir}/{img_path}"
			target_img = img_data
			
		#img = cv2.imread("C:/Users/USER/Pictures/a.jpg")
		img = target_img
		
		# Convert a BGR image to a grayscale image
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Set the maximum value for the histogram
		max_val = 256

		# Compute a gray level histogram before equalization
		#hist = cv2.calcHist([gray_img], [0], None, [max_val], [0, max_val])

		# Implementing histogram equalization algorithm
		G = max_val
		M, N = gray_img.shape
		H = np.zeros(G, dtype=int)

		for i in range(M):
			for j in range(N):
				intensity = gray_img[i, j]
				H[intensity] += 1

		g_min = next(i for i, val in enumerate(H) if val > 0)
		H_c = np.zeros(G, dtype=int)
		H_c[0] = H[0]

		for i in range(1, G):
			H_c[i] = H_c[i - 1] + H[i]

		H_min = H_c[g_min]
		T = np.zeros(G, dtype=int)

		for i in range(G):
			T[i] = round((H_c[i] - H_min) / (M * N - H_min) * (G - 1))

		equalized_img = np.zeros_like(gray_img, dtype=np.uint8)

		for i in range(M):
			for j in range(N):
				equalized_img[i, j] = T[gray_img[i, j]]

		# Compute a gray level histogram after equalization
		equalized_hist = cv2.calcHist([equalized_img], [0], None, [max_val], [0, max_val])
		
		# Draw the histograms
		plt.bar(range(max_val), equalized_hist.flatten())
		
		# Disable x-ticks & y-ticks
		plt.xticks([])
		plt.yticks([])

		# Add labels and title
		plt.xlabel("Pixel intensity")
		plt.ylabel("Number of pixels (normalized)")
		plt.title("Gray level histogram")

		# Save the chart
		plt.savefig(equalized_grayHist_path, dpi=200)
		
		equalized_img = cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR)
		return equalized_grayHist_path, equalized_img
	
	def draw_and_save_grayHist(self, save_dir, img_data=None, img_path=""):
		if img_data is None :
			grayHist_path = f"{save_dir}/gray_level_histogram.png"
			if os.path.exists(grayHist_path):
				os.remove(grayHist_path)
			target_img = self.img
		else:
			grayHist_path = f"{save_dir}/{img_path}"
			target_img = img_data

		# Resize the image to reduce the complexity of computation afterward
		'''
		h, w = self.img.shape[:2]
		new_height = 128
		new_width = int(w / h * 128)
		resized_img = cv2.resize(self.img, (new_width, new_height))
		'''

		# Convert a BGR image to a grayscale image
		gray_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

		# Set the maximum value for the histogram
		max_val = 256

		# Compute a gray level histogram
		hist = cv2.calcHist([gray_img], [0], None, [max_val], [0, max_val])

		# Normalize the number values via min-max normalization
		# and convert these values into the percentage values
		hist = 100 * ((hist - hist.min()) / (hist.max() - hist.min()))

		# Convert the data-structure of the values of gray level histogram
		list_of_num_of_pixels = [int(e[0]) for e in hist.tolist()]
		#print(list_of_num_of_pixels, f"=> # of values: {len(list_of_num_of_pixels)}", sep='\n')
		#print([str(e) for e in range(0,256+1)])
		##############

		# Sample data
		categories = [str(e) for e in range(0, max_val)]
		values = list_of_num_of_pixels

		# Create a bar chart
		#colors = [(0.2, 0.4, 0.6), (0.8, 0.2, 0.4), (0.1, 0.6, 0.2)]
		colors = [(0.2, 0.4, 0.6)] * len(values)
		plt.bar(categories, values, color=colors)

		# Set a larger interval for x-axis to prevent overlap among the categories
		#interval = 20  # Set the interval you want
		#plt.xticks(np.arange(0, max_val+1, interval))

		# Disable x-ticks & y-ticks
		plt.xticks([])
		plt.yticks([])

		# Add labels and title
		plt.xlabel("Pixel intensity")
		plt.ylabel("Number of pixels (normalized)")
		plt.title("Gray level histogram")

		# Save the chart
		plt.savefig(grayHist_path, dpi=200)

		# Display the chart
		#plt.show()
		return grayHist_path

	def generate_gaussian_noise(self, img, sigma=50, G=200):
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
		noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_GRAY2BGR)

		return noisy_img, noise

	def generate_saltAndPepper_noise(self, img, salt_prob=0.03, pepper_prob=0.03):
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

	def add_padding(self, img, padding_width):
		img = np.array(img)

		# Array of zeros of shape (img + padding_width)
		padded_img = np.zeros(shape=(
			img.shape[0] + padding_width * 2,  # Multiply with two because we need padding on all sides
			img.shape[1] + padding_width * 2
		))

		padded_img[padding_width:-padding_width, padding_width:-padding_width] = img

		return padded_img

	def convolve(self, img, kernel):
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Create an empty output image
		convolved_img = np.zeros_like(gray_img)

		# Flip the kernel horizontally and vertically (convolution theorem)
		kernel = np.flipud(np.fliplr(kernel))

		# Define padding size (e.g. "1" for a 3x3 kernel)
		padding = len(kernel) // 2

		# Apply zero-padding to the image
		#padded_img = np.pad(img, ((padding, padding), (padding, padding)), mode="constant")
		padded_img = self.add_padding(gray_img, padding)

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

		convolved_img = cv2.cvtColor(convolved_img, cv2.COLOR_GRAY2BGR)
		return convolved_img

class MainWindow(QWidget):
	def __init__(self):
		super().__init__()
		self.initUI()
		self.img_path = "X"
		self.LOCAL_IMG_DIR = "images"

		# Create the directory to save images
		os.mkdir(self.LOCAL_IMG_DIR) if not os.path.exists(self.LOCAL_IMG_DIR) else None

		# If there are previous images in the directory, delete them
		self.remove_images()

		self.showMaximized()
		self.image_process = ImageProcess()

	def open_noise_window(self):
		self.noise_window = NoiseGenerateWindow()
		self.noise_window.confirm_signal.connect(self.confirm_noise_received)
		self.noise_window.show()

	def open_convolve_window(self):
		self.convolve_window = ConvolveWindow()
		self.convolve_window.confirm_signal.connect(self.confirmation_conv_received)
		self.convolve_window.show()

	def confirm_noise_received(self): # receive signal from NoiseGenerateWindow
		raw_img = cv2.imread(self.img_path)

		if self.noise_window.btnGaussainNoise.isChecked():
			# Get typed standard deviation
			try:
				sigma_val = float(self.noise_window.standard_deviation.text())

			except ValueError:
				#print("輸入格式有誤（請輸入數字）")
				QMessageBox.information(self, "錯誤訊息", "輸入格式有誤（請在 [標準差] 旁的輸入文字框中輸入數字）", QMessageBox.Ok)

			else:
				''' Upper row: raw image, noise, and noisy image '''
				noisy_img, noise = self.image_process.generate_gaussian_noise(raw_img, sigma_val)

		else:
			noisy_img, noise = self.image_process.generate_saltAndPepper_noise(raw_img)

		self.update_specific_image((0,1), noise)
		self.update_specific_image((0,2), noisy_img)

		''' Lower row: histogram figures of the upper-row images '''
		lower_fig_paths = ["raw_img.png", "noise.png", "noisy_img.png"]

		saved_path = self.image_process.draw_and_save_grayHist(self.LOCAL_IMG_DIR, raw_img, lower_fig_paths[0])
		img = cv2.imread(saved_path)
		self.update_specific_image((1,0), img)

		saved_path = self.image_process.draw_and_save_grayHist(self.LOCAL_IMG_DIR, noise, lower_fig_paths[1])
		img = cv2.imread(saved_path)
		self.update_specific_image((1,1), img)

		saved_path = self.image_process.draw_and_save_grayHist(self.LOCAL_IMG_DIR, noisy_img, lower_fig_paths[2])
		img = cv2.imread(saved_path)
		self.update_specific_image((1,2), img)

	def confirmation_conv_received(self): # receive signal from ConvolveWindow
		raw_img = cv2.imread(self.img_path)

		tmp = self.convolve_window.kernel
		if tmp is not None:
			kernel = tmp
			#print("kernel:", kernel, sep='\n')
			kernel = np.array(kernel)

			QMessageBox.information(self, "提示訊息", "卷積運算執行中...", QMessageBox.Ok)

			convolved_img = self.image_process.convolve(raw_img, kernel)

			self.update_specific_image((0,1), convolved_img)

			QMessageBox.information(self, "提示訊息", f"您輸入的卷積核：\n{kernel}", QMessageBox.Ok)

		else:
			QMessageBox.information(self, "錯誤訊息", "請重新選擇卷積核尺寸，並在表格中輸入數值", QMessageBox.Ok)

	def remove_images(self):
		existed_img_paths = [f"{self.LOCAL_IMG_DIR}/{e}" for e in os.listdir(self.LOCAL_IMG_DIR)]
		[os.remove(e) for e in existed_img_paths]

	def create_table(self):
		table = QTableWidget()
		table.setColumnCount(3)
		table.setRowCount(2)

		#table.setHorizontalHeaderLabels(["圖片1","圖片2", "圖片3"])
		table.setEditTriggers(QAbstractItemView.NoEditTriggers)

		# Set the size of each item
		x = 384 ; table.setIconSize(QSize(x, x))

		table.verticalHeader().setVisible(False) # hide the row headers
		table.horizontalHeader().setVisible(False) # hide the column headers

		for i in range(3):
		   table.setColumnWidth(i, x)

		for i in range(2):
		   table.setRowHeight(i, x)

		return table

	def update_left_image(self):
		# If there are previous images in the directory, delete them
		self.remove_images()

		# Clear the left & right image
		self.table.setItem(0,0,QTableWidgetItem())
		self.table.setItem(0,1,QTableWidgetItem())
		self.table.setItem(0,2,QTableWidgetItem())
		self.table.setItem(1,0,QTableWidgetItem())
		self.table.setItem(1,1,QTableWidgetItem())
		self.table.setItem(1,2,QTableWidgetItem())
		self.lbl_imgFormatType.setText("未知")
		self.lbl_imgSizeType.setText("未知")

		try:
			# Open an image via file opener
			img_path, _  = QFileDialog.getOpenFileName(self, "開啟檔案", "C:/","Image files (*.jpg *.bmp *.ppm *.png)")
		except:
			pass

		if os.path.exists(img_path):
			try:
				# Record and show the extension of the original image
				self.extension = img_path.split('.')[-1]
				self.lbl_imgFormatType.setText(self.extension.upper())

				# Record filename of the original image
				#self.img_path = img_path
				self.img_path = f"{self.LOCAL_IMG_DIR}/original.{self.extension}"

				# Copy the image from path: `img_path` to path: `self.img_path`
				# in order to solve the problem that
				# OpenCV cannot read image with a non-Engliah location
				shutil.copy(img_path, self.img_path)

				# Set the original image in the instance of ImageProcess
				self.image_process.set_image(self.img_path)

				# Show the image size
				#h, w = original_img.shape[:2]
				h, w = self.image_process.img.shape[:2]
				self.lbl_imgSizeType.setText(f"{w} (寬) x {h} (高)")

				# Save the original image to local directory
				#cv2.imwrite(self.img_path, original_img)

				# Update the left item with original image in the table
				item = QTableWidgetItem()
				item.setTextAlignment(Qt.AlignHCenter)
				item.setFlags(Qt.ItemIsEnabled) # select the image when it is clicked
				icon = QIcon(QPixmap(self.img_path))
				item.setIcon(QIcon(icon))
				self.table.setItem(0,0,item)
			except:
				pass

	def update_right_image(self, option):
		if option == "2a":
			target_img = self.rotated_img
		elif option == "2b":
			target_img = self.grayHist_img

		item = QTableWidgetItem()
		item.setTextAlignment(Qt.AlignHCenter)
		item.setFlags(Qt.ItemIsEnabled) # select the image when it is clicked

		h, w, c = target_img.shape
		bytes_per_line = 3 * w
		target_img_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
		q_img = QImage(target_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

		pixmap = QPixmap.fromImage(q_img)
		icon = QIcon(pixmap)

		#icon = QIcon(QPixmap(self.rotated_img_path))
		item.setIcon(QIcon(icon))
		self.table.setItem(0,1,item)

	def initUI(self):
		self.setWindowTitle("AIP-NTUST_M11125014")
		self.resize(1250,630+100)

		self.le_rotation_angle = QLineEdit()
		self.le_rotation_angle.setFixedWidth(60)

		# Add a functional button
		sub_layout_0 = QHBoxLayout()

		'''
		lbl_functions = QLabel("1.請選擇功能")
		lbl_functions.setFont(QFont("Arial", 10))
		sub_layout_0.addWidget(lbl_functions)
		'''

		upload_img_btn = QPushButton("1. 載入圖片")
		upload_img_btn.clicked.connect(self.update_left_image)
		upload_img_btn.setFixedSize(150, 30)
		sub_layout_0.addWidget(upload_img_btn, 0, Qt.AlignLeft)

		lbl_imgFormat = QLabel("\t影像格式：")
		lbl_imgFormat.setFont(QFont("Arial", 10))
		sub_layout_0.addWidget(lbl_imgFormat, 0, Qt.AlignLeft)

		lbl_imgFormatType = QLabel("未知")
		lbl_imgFormatType.setFont(QFont("Arial", 10))
		self.lbl_imgFormatType = lbl_imgFormatType
		sub_layout_0.addWidget(lbl_imgFormatType, 0, Qt.AlignLeft)

		lbl_imgSize = QLabel("\t影像大小：")
		lbl_imgSize.setFont(QFont("Arial", 10))
		sub_layout_0.addWidget(lbl_imgSize, 0, Qt.AlignLeft)

		lbl_imgSizeType = QLabel("未知")
		lbl_imgSizeType.setFont(QFont("Arial", 10))
		self.lbl_imgSizeType = lbl_imgSizeType
		sub_layout_0.addWidget(lbl_imgSizeType, 1000, Qt.AlignLeft)

		######################################
		''' Arguments '''

		sub_layout_1 = QFormLayout()
		sub_layout_1.addRow("旋轉角度：", self.le_rotation_angle)

		######################################
		''' Buttons '''

		sub_layout_2 = QHBoxLayout()

		btn_rotateImg = QPushButton("2a. 旋轉圖片")
		btn_rotateImg.setFixedSize(150, 30)
		btn_rotateImg.clicked.connect(self.rotate_image_button_clicked)
		sub_layout_2.addWidget(btn_rotateImg)

		lbl_Btn1Tab = QLabel("\t")
		lbl_Btn1Tab.setFont(QFont("Arial", 10))
		sub_layout_2.addWidget(lbl_Btn1Tab, 0, Qt.AlignLeft)

		btn_drawHist = QPushButton("2b. 繪製直方圖")
		btn_drawHist.setFixedSize(150, 30)
		btn_drawHist.clicked.connect(self.draw_hist_button_clicked)
		sub_layout_2.addWidget(btn_drawHist, 0, Qt.AlignLeft)

		lbl_Btn2Tab = QLabel("\t")
		lbl_Btn2Tab.setFont(QFont("Arial", 10))
		sub_layout_2.addWidget(lbl_Btn2Tab, 0, Qt.AlignLeft)

		btn_genNoise = QPushButton("2c. 產生雜訊")
		btn_genNoise.setFixedSize(150, 30)
		btn_genNoise.clicked.connect(self.gen_noise_button_clicked)
		sub_layout_2.addWidget(btn_genNoise, 0, Qt.AlignLeft)

		lbl_Btn3Tab = QLabel("\t")
		lbl_Btn3Tab.setFont(QFont("Arial", 10))
		sub_layout_2.addWidget(lbl_Btn3Tab, 0, Qt.AlignLeft)

		btn_genNoise = QPushButton("2d. 卷積運算")
		btn_genNoise.setFixedSize(150, 30)
		btn_genNoise.clicked.connect(self.convolve_button_clicked)
		sub_layout_2.addWidget(btn_genNoise, 0, Qt.AlignLeft)
		
		lbl_Btn4Tab = QLabel("\t")
		lbl_Btn4Tab.setFont(QFont("Arial", 10))
		sub_layout_2.addWidget(lbl_Btn4Tab, 0, Qt.AlignLeft)

		btn_HE = QPushButton("2e. 直方圖均化")
		btn_HE.setFixedSize(150, 30)
		btn_HE.clicked.connect(self.HE_button_clicked)
		sub_layout_2.addWidget(btn_HE, 1000, Qt.AlignLeft)

		######################################

		sub_layout_3 = QVBoxLayout()

		# Add a table to the layout
		self.table = self.create_table()
		sub_layout_3.addWidget(self.table)

		######################################

		# Define the primary layout
		main_layout = QVBoxLayout()

		# Add wigets about "functional button" to main_layout
		w0 = QWidget()
		w0.setLayout(sub_layout_0)
		main_layout.addWidget(w0)

		# Add wigets about "rotation angle" to main_layout
		w1 = QWidget()
		w1.setLayout(sub_layout_1)
		main_layout.addWidget(w1)

		# Add wigets about "XXX" to main_layout
		w2 = QWidget()
		w2.setLayout(sub_layout_2)
		main_layout.addWidget(w2)

		# Add wigets about "images" to main_layout
		w3 = QWidget()
		w3.setLayout(sub_layout_3)
		main_layout.addWidget(w3)
		self.setLayout(main_layout)

	def rotate_image_button_clicked(self):
		if self.img_path == "X":
			QMessageBox.information(self, "錯誤訊息", "請選擇圖片", QMessageBox.Ok)
		else:
			# Get typed rotation angle
			try:
				rotation_angle = float(self.le_rotation_angle.text())

			except ValueError:
				#print("輸入格式有誤（請輸入數字）")
				QMessageBox.information(self, "錯誤訊息", "輸入格式有誤（請輸入數字）", QMessageBox.Ok)

			else:
				# Save typed rotation angle
				self.rotation_angle = rotation_angle

				# Rotate the original image
				# Read the uploaded image
				self.rotated_img = self.image_process.rotate(self.rotation_angle)

				# Update `rotated_img_path`
				rotated_img_path = '.'.join(self.img_path.split('.')[:-1]) + "_v2" + ".bmp"
				cv2.imwrite(rotated_img_path, self.rotated_img)

				# Update the right image with a rotated imag
				self.update_right_image("2a")

	def draw_hist_button_clicked(self):
		if self.img_path == "X":
			QMessageBox.information(self, "錯誤訊息", "請選擇圖片", QMessageBox.Ok)

		else:
			grayHist_path = self.image_process.draw_and_save_grayHist(self.LOCAL_IMG_DIR)

			self.grayHist_img = cv2.imread(grayHist_path)

			# Update the right image with a rotated imag
			self.update_right_image("2b")

	def update_specific_image(self, loc, target_img):
		item = QTableWidgetItem()
		item.setTextAlignment(Qt.AlignHCenter)
		item.setFlags(Qt.ItemIsEnabled) # select the image when it is clicked

		h, w, c = target_img.shape
		bytes_per_line = 3 * w
		target_img_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
		q_img = QImage(target_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

		pixmap = QPixmap.fromImage(q_img)
		icon = QIcon(pixmap)

		#icon = QIcon(QPixmap(self.rotated_img_path))
		item.setIcon(QIcon(icon))
		self.table.setItem(loc[0], loc[1],item)

	def gen_noise_button_clicked(self):
		if self.img_path == "X":
			QMessageBox.information(self, "錯誤訊息", "請選擇圖片", QMessageBox.Ok)
		else:
			self.open_noise_window()

	def convolve_button_clicked(self):
		if self.img_path == "X":
			QMessageBox.information(self, "錯誤訊息", "請選擇圖片", QMessageBox.Ok)
		else:
			self.open_convolve_window()
			
	def HE_button_clicked(self):
		if self.img_path == "X":
			QMessageBox.information(self, "錯誤訊息", "請選擇圖片", QMessageBox.Ok)
		else:
			# Draw a normal histogram
			saved_path = self.image_process.draw_and_save_grayHist(self.LOCAL_IMG_DIR)
			hist = cv2.imread(saved_path)
			self.update_specific_image((1,0), hist)
			
			# Draw a histogram with Histogram Equalization
			saved_path, equalized_img = self.image_process.draw_and_save_grayHist_with_equalization(self.LOCAL_IMG_DIR)
			self.update_specific_image((0,1), equalized_img)
			equalized_hist = cv2.imread(saved_path)
			self.update_specific_image((1,1), equalized_hist)

if __name__ == "__main__":
	app = QApplication(sys.argv)
	main_win = MainWindow()
	main_win.show()
	sys.exit(app.exec_())
