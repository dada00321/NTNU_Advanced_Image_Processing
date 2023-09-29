"""
Advanced Image Processing - HW1

> Title of the main window: `AIP-NTUST_M11125014`
> File name of the .exe file: `HW1-NTUST_M11125014.exe`
> Input image extensions (at least): JPG, BMP, PPM
> Output: (No special restrictions)
> Function-1: Read image
> Function-2: Rotate image
> Function-3: Identify the image format & image size
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

class ImageProcess():
	def set_image(self, img_path):
		self.img = cv2.imread(img_path)

	def rotate(self, rotation_angle):
		# Rotate the image
		rotated_img = imutils.rotate_bound(self.img, rotation_angle)
		return rotated_img

	def draw_and_save_grayHist(self, save_dir):
		grayHist_path = f"{save_dir}/gray_level_histogram.png"
		if os.path.exists(grayHist_path):
			os.remove(grayHist_path)

		# Resize the image to reduce the complexity of computation afterward
		'''
		h, w = self.img.shape[:2]
		new_height = 128
		new_width = int(w / h * 128)
		resized_img = cv2.resize(self.img, (new_width, new_height))
		'''

		# Convert a BGR image to a grayscale image
		gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

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

	def remove_images(self):
		existed_img_paths = [f"{self.LOCAL_IMG_DIR}/{e}" for e in os.listdir(self.LOCAL_IMG_DIR)]
		[os.remove(e) for e in existed_img_paths]

	def create_table(self):
		table = QTableWidget()
		table.setColumnCount(2)
		table.setRowCount(1)

		table.setHorizontalHeaderLabels(["圖片1","圖片2"])
		table.setEditTriggers(QAbstractItemView.NoEditTriggers)

		x = 600
		table.setIconSize(QSize(x, x))

		table.verticalHeader().setVisible(False) # hide the row headers

		for i in range(2):
		   table.setColumnWidth(i, x)

		table.setRowHeight(0, x)

		return table

	def update_left_image(self):
		# If there are previous images in the directory, delete them
		self.remove_images()

		# Clear the left & right image
		self.table.setItem(0,0,QTableWidgetItem())
		self.table.setItem(0,1,QTableWidgetItem())
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
		sub_layout_2.addWidget(btn_drawHist, 1000, Qt.AlignLeft)

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

if __name__ == "__main__":
	app = QApplication(sys.argv)
	main_win = MainWindow()
	main_win.show()
	sys.exit(app.exec_())
