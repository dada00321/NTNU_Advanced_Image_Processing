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

class ImageProcess():
	def rotate(self, img, rotation_angle):
		# Rotate the image
		rotated_img = imutils.rotate_bound(img, rotation_angle)
		return rotated_img

class MainWindow(QWidget):
	def __init__(self):
		super().__init__()
		self.initUI()
		self.img_path = "X"
		self.local_img_path = "images"
		os.mkdir(self.local_img_path) if not os.path.exists(self.local_img_path) else None
		self.showMaximized()

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
		# Open an image via file opener
		img_path, _  = QFileDialog.getOpenFileName(self, "開啟檔案", "C:/","Image files (*.jpg *.bmp *.ppm)")

		# Clear the right image
		item = QTableWidgetItem()
		self.table.setItem(0,1,item)

		# Record and show the extension of the original image
		self.extension = img_path.split('.')[-1]
		self.lbl_imgFormatType.setText(self.extension.upper())

		# Record filename of the original image
		#self.img_path = img_path
		self.img_path = f"{self.local_img_path}/original.{self.extension}"

		# Copy the image from path: `img_path` to path: `self.img_path`
		# in order to solve the problem that
		# OpenCV cannot read image with a non-Engliah location
		shutil.copy(img_path, self.img_path)

		# Read the uploaded image
		"""
		if self.extension == "ppm":
			sk_img = skimage.io.imread(self.img_path) # Read the PPM image
			cv_img = skimage.img_as_ubyte(sk_img)
			original_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		else:
			original_img = cv2.imread(self.img_path)
		"""
		original_img = cv2.imread(self.img_path)
		# Show the image size
		h, w = original_img.shape[:2]
		self.lbl_imgSizeType.setText(f"{w} (寬) x {h} (高)")

		# Resize the image (in order to fit the window size)
		'''
		h, w = original_img.shape[:2]
		x = 935
		if h > w:
			new_h, new_w = x, int(w * (x/h))
		else:
			new_h, new_w = int(h * (x/w)), x
		original_img = cv2.resize(original_img, (new_w, new_h))
		'''

		# Save the original image to local directory
		cv2.imwrite(self.img_path, original_img)

		# Update the left item with original image in the table
		item = QTableWidgetItem()
		item.setTextAlignment(Qt.AlignHCenter)
		item.setFlags(Qt.ItemIsEnabled) # select the image when it is clicked
		icon = QIcon(QPixmap(self.img_path))
		item.setIcon(QIcon(icon))
		self.table.setItem(0,0,item)

	def update_right_image(self):
		rotated_img_path = "1.jpg"
		item = QTableWidgetItem()
		item.setTextAlignment(Qt.AlignHCenter)
		item.setFlags(Qt.ItemIsEnabled) # select the image when it is clicked

		h, w, c = self.rotated_img.shape
		bytes_per_line = 3 * w
		rotated_img_rgb = cv2.cvtColor(self.rotated_img, cv2.COLOR_BGR2RGB)
		q_img = QImage(rotated_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
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
		#---
		'''
		lbl_functions = QLabel("1.請選擇功能")
		lbl_functions.setFont(QFont("Arial", 10))
		sub_layout_0.addWidget(lbl_functions)
		'''
		#---
		upload_img_btn = QPushButton("載入圖片")
		upload_img_btn.clicked.connect(self.update_left_image)
		upload_img_btn.setFixedSize(150, 30)
		sub_layout_0.addWidget(upload_img_btn, 0, Qt.AlignLeft)
		#---
		lbl_imgFormat = QLabel("\t影像格式：")
		lbl_imgFormat.setFont(QFont("Arial", 10))
		sub_layout_0.addWidget(lbl_imgFormat, 0, Qt.AlignLeft)
		#---
		lbl_imgFormatType = QLabel("未知")
		lbl_imgFormatType.setFont(QFont("Arial", 10))
		self.lbl_imgFormatType = lbl_imgFormatType
		sub_layout_0.addWidget(lbl_imgFormatType, 0, Qt.AlignLeft)
		#---
		lbl_imgSize = QLabel("\t影像大小：")
		lbl_imgSize.setFont(QFont("Arial", 10))
		sub_layout_0.addWidget(lbl_imgSize, 0, Qt.AlignLeft)
		#---
		lbl_imgSizeType = QLabel("未知")
		lbl_imgSizeType.setFont(QFont("Arial", 10))
		self.lbl_imgSizeType = lbl_imgSizeType
		sub_layout_0.addWidget(lbl_imgSizeType, 1000, Qt.AlignLeft)

		######################################

		sub_layout_1 = QFormLayout()
		sub_layout_1.addRow("旋轉角度：", self.le_rotation_angle)

		btn_rotateImg = QPushButton("旋轉圖片")
		btn_rotateImg.setFixedSize(150, 30)
		btn_rotateImg.clicked.connect(self.rotate_image_button_clicked)
		sub_layout_1.addRow(btn_rotateImg)

		######################################

		sub_layout_2 = QVBoxLayout()

		# Add a table to the layout
		self.table = self.create_table()
		sub_layout_2.addWidget(self.table)

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

		# Add wigets about "images" to main_layout
		w2 = QWidget()
		w2.setLayout(sub_layout_2)
		main_layout.addWidget(w2)
		self.setLayout(main_layout)

	def rotate_image_button_clicked(self):
		if self.img_path == "X":
			QMessageBox.information(self, "錯誤訊息", "請選擇圖片", QMessageBox.Ok)

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
			"""
			if self.extension == "ppm":
				sk_img = skimage.io.imread(self.img_path) # Read the PPM image
				cv_img = skimage.img_as_ubyte(sk_img)
				img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
			else:
				img = cv2.imread(self.img_path)
			"""
			img = cv2.imread(self.img_path)
			self.rotated_img = ImageProcess().rotate(img, self.rotation_angle)

			# Update `rotated_img_path`
			rotated_img_path = '.'.join(self.img_path.split('.')[:-1]) + "_v2" + ".bmp"
			cv2.imwrite(rotated_img_path, self.rotated_img)

			# Update the right image with a rotated imag
			self.update_right_image()

if __name__ == "__main__":
	app = QApplication(sys.argv)
	example = MainWindow()
	example.show()
	sys.exit(app.exec_())
