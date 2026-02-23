import sys
import os
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from colorize import colorize_image


class ColorItApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ABLAZE - Image Colorizer")
        self.setGeometry(200, 100, 900, 600)
        self.setStyleSheet("background-color: #f5f5f5;")

        self.image_path = None
        self.colorized_image = None

        # ---------- Layout Setup ----------
        layout = QtWidgets.QGridLayout(self)

        # Input image label
        self.input_label = QtWidgets.QLabel("Input Image")
        self.input_label.setAlignment(QtCore.Qt.AlignCenter)
        self.input_label.setStyleSheet("border: 1px solid #ccc; background: white;")
        layout.addWidget(self.input_label, 0, 0, 1, 1)

        # Output image label
        self.output_label = QtWidgets.QLabel("Colorized Image")
        self.output_label.setAlignment(QtCore.Qt.AlignCenter)
        self.output_label.setStyleSheet("border: 1px solid #ccc; background: white;")
        layout.addWidget(self.output_label, 0, 1, 1, 1)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()

        self.open_button = QtWidgets.QPushButton("Open Image")
        self.open_button.clicked.connect(self.open_image)
        btn_layout.addWidget(self.open_button)

        self.colorize_button = QtWidgets.QPushButton("Colorize (Predict)")
        self.colorize_button.clicked.connect(self.colorize)
        btn_layout.addWidget(self.colorize_button)

        self.save_button = QtWidgets.QPushButton("Save Result")
        self.save_button.clicked.connect(self.save_result)
        btn_layout.addWidget(self.save_button)

        layout.addLayout(btn_layout, 1, 0, 1, 2)

        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Open image
    # ------------------------------------------------------------------
    def open_image(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Grayscale Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.image_path = file_path
            pixmap = QtGui.QPixmap(file_path).scaled(
                400, 400, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            self.input_label.setPixmap(pixmap)
            self.output_label.clear()
            self.output_label.setText("Colorized Image")
            print(f"Loaded image: {file_path}")

    # ------------------------------------------------------------------
    # Colorize and enhance image
    # ------------------------------------------------------------------
    def colorize(self):
        if not self.image_path:
            QtWidgets.QMessageBox.warning(self, "Error", "Please open an image first.")
            return

        try:
            print("Colorizing and enhancing...")
            colorized_img = colorize_image(self.image_path)

            if colorized_img is None:
                raise ValueError("Colorization failed - no output image produced.")

            print("Colorized image shape:", colorized_img.shape, "dtype:", colorized_img.dtype)

            # Save for later use
            self.colorized_image = colorized_img

            # Convert to Qt image
            rgb_image = cv2.cvtColor(colorized_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(
                rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
            )

            # Display
            pixmap = QtGui.QPixmap.fromImage(qt_image)
            pixmap = pixmap.scaled(
                400, 400, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            self.output_label.setPixmap(pixmap)
            print("Colorization and enhancement complete!")

        except Exception as e:
            print("Failed to colorize:", e)
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to colorize:\n{e}")

    # ------------------------------------------------------------------
    # Save final result
    # ------------------------------------------------------------------
    def save_result(self):
        if self.colorized_image is None:
            QtWidgets.QMessageBox.warning(self, "Error", "No colorized image to save.")
            return

        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Colorized Image", "", "JPEG (*.jpg);;PNG (*.png)"
        )
        if save_path:
            cv2.imwrite(save_path, self.colorized_image)
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved to:\n{save_path}")
            print(f"Image saved to {save_path}")


# ----------------------------------------------------------------------
# Run Application
# ----------------------------------------------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ColorItApp()
    window.show()
    sys.exit(app.exec_())
