import os
import warnings
from PIL import Image
warnings.filterwarnings('ignore')

import tensorflow as tf
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import Model

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QMovie, QFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image

import numpy as np
import cv2
from win32com.client import Dispatch
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices


# Function to handle text-to-speech
def speak(str1):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(str1)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 700)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(0, 0, 800, 700))
        self.frame.setStyleSheet("background-color: #B0D4E3; border-radius: 10px;")
        self.frame.setObjectName("frame")

        # GIF setup
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(100, 50, 600, 400))
        self.label.setText("")
        self.gif = QMovie("picture.gif")  
        self.label.setMovie(self.gif)
        self.gif.start()
        self.label.setObjectName("label")

        # Title Label
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(100, 480, 600, 50))
        font = QtGui.QFont("Arial", 24, QtGui.QFont.Bold)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")

        # Buttons
        self.pushButton = QtWidgets.QPushButton(self.frame)
        self.pushButton.setGeometry(QtCore.QRect(50, 600, 200, 40))
        self.pushButton.setText("Upload Image")
        self.pushButton.setStyleSheet("background-color: #DF582C; color: white; font-size: 14px; border-radius: 10px;")
        self.pushButton.setObjectName("pushButton")

        self.pushButton_2 = QtWidgets.QPushButton(self.frame)
        self.pushButton_2.setGeometry(QtCore.QRect(300, 600, 200, 40))
        self.pushButton_2.setText("Predict")
        self.pushButton_2.setStyleSheet("background-color: #DF582C; color: white; font-size: 14px; border-radius: 10px;")
        self.pushButton_2.setObjectName("pushButton_2")

        self.pushButton_3 = QtWidgets.QPushButton(self.frame)
        self.pushButton_3.setGeometry(QtCore.QRect(550, 600, 200, 40))
        self.pushButton_3.setText("Download Report")
        self.pushButton_3.setStyleSheet("background-color: #DF582C; color: white; font-size: 14px; border-radius: 10px;")
        self.pushButton_3.setObjectName("pushButton_3")

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(self.upload_image)
        self.pushButton_2.clicked.connect(self.predict_result)
        self.pushButton_3.clicked.connect(self.download_report)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Pneumonia Detection"))
        self.label_2.setText(_translate("MainWindow", "Pneumonia Detection System"))

    def upload_image(self):
        filename = QFileDialog.getOpenFileName()
        self.image_path = filename[0]
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            pixmap = pixmap.scaled(self.label.width(), self.label.height(), QtCore.Qt.KeepAspectRatio)
            self.label.setPixmap(pixmap)

    def predict_result(self):
        global result, highlighted_image

        if not hasattr(self, 'image_path') or not self.image_path:
            QMessageBox.warning(None, "Warning", "Please upload an image first!")
            return

        model = load_model('chest_xray_.h5') 
        img = image.load_img(self.image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        prediction = model.predict(img_array)
        result = prediction

        if prediction[0][0] > 0.5:
            result_text = "Result: Normal"
            color = "green"
            highlighted_image = None
        else:
            result_text = "Result: Affected by Pneumonia"
            color = "red"
            highlighted_image = self.generate_grad_cam(model, img_array)

            # Redirect to website for booking an appointment
            QDesktopServices.openUrl(QUrl("https://medipulse-parth-tyagi.netlify.app/"))

        self.label_2.setText(result_text)
        self.label_2.setStyleSheet(f"color: {color}; font-weight: bold;")

        speak(result_text)

    def generate_grad_cam(self, model, img_array):
        last_conv_layer = model.get_layer("block5_conv3")
        grad_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  
        conv_outputs = conv_outputs.numpy()[0]
        pooled_grads = pooled_grads.numpy()

        for i in range(pooled_grads.shape[0]):
            conv_outputs[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        img = cv2.imread(self.image_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        num_circles = 5
        heatmap_indices = np.unravel_index(np.argsort(heatmap.ravel())[-num_circles:], heatmap.shape)

        for idx in range(num_circles):
            y, x = heatmap_indices[0][idx], heatmap_indices[1][idx]
            center = (x, y)
            radius = np.random.randint(30, 70)
            cv2.circle(superimposed_img, center, radius, (0, 0, 255), 5)

        output_path = "highlighted_image.jpg"
        cv2.imwrite(output_path, superimposed_img)
        return output_path

    def download_report(self):
        if result is None or np.all(result == 0):
            QMessageBox.warning(None, "Warning", "Please make a prediction first!")
            return

        report_path = QFileDialog.getSaveFileName(None, "Save Report", "", "PDF Files (*.pdf)")[0]
        if report_path:
            doc = SimpleDocTemplate(report_path, pagesize=letter)
            elements = []

            # Sample Styles
            styles = getSampleStyleSheet()
            title_style = styles['Title']
            title_style.fontSize = 24
            title_style.alignment = 1  # Center alignment

            # Cover Page
            hospital_logo = "C:\\Users\\hp\\Desktop\\Pneumonia_Detection\\Pneumonia_Detection\\Chest_x_ray_Detection-master\\logo.png"
            if os.path.exists(hospital_logo):
                logo_image = Image(hospital_logo, width=200, height=100)  # Increased width
                elements.append(logo_image)
                elements.append(Spacer(1, 12))

            title = "Pneumonia Detection Report"
            elements.append(Paragraph(title, title_style))
            elements.append(Spacer(1, 24))

            # Patient Information Section
            patient_style = ParagraphStyle(name='patient_info', fontName='Helvetica', fontSize=12, spaceAfter=12)
            patient_info = """
            <b>Patient Name:</b> John Doe<br/>
            <b>Age:</b> 45<br/>
            <b>Gender:</b> Male<br/>
            <b>Medical History:</b> No known allergies, No chronic diseases<br/>
            <b>Contact:</b> 123-456-7890<br/>
            <b>Consulting Doctor:</b> Dr. Sarah Smith<br/>
            <b>Report Date:</b> {0}
            """.format("2024-11-15")  # Add the current date
            patient_paragraph = Paragraph(patient_info, patient_style)
            elements.append(patient_paragraph)
            elements.append(Spacer(1, 12))

            # Prediction Result Section
            prediction_style = styles["Normal"]
            prediction_text = """
            <b>Prediction Result:</b> {0}<br/>
            <b>Diagnosis:</b> {1}
            """.format("Pneumonia Detected" if result[0][0] < 0.5 else "Normal",
                    "The patient has pneumonia and should consult a doctor for further treatment." if result[0][0] < 0.5 else "No signs of pneumonia")
            prediction_paragraph = Paragraph(prediction_text, prediction_style)
            elements.append(prediction_paragraph)
            elements.append(Spacer(1, 12))

            # Add space for Grad-CAM Image Section
            if highlighted_image:
                grad_cam_img = Image(highlighted_image, width=500, height=350)
                elements.append(grad_cam_img)
                elements.append(Spacer(1, 12))

            # Doctor's Recommendation Section
            doctor_recommendation = """
            <b>Doctor's Recommendation:</b><br/>
            If pneumonia is detected, it's crucial to follow prescribed medication and rest.<br/>
            Ensure adequate hydration and regular follow-ups with the healthcare provider.<br/>
            Avoid strenuous activities and take proper precautions to prevent spreading the disease.<br/><br/>
            <b>Next Steps:</b><br/>
            1. Immediate consultation with a pulmonologist.<br/>
            2. Regular monitoring and follow-up tests.<br/>
            3. Consider a chest CT scan for further diagnosis.<br/>
            4. Maintain a healthy diet to support lung health.<br/>
            """
            recommendation_paragraph = Paragraph(doctor_recommendation, prediction_style)
            elements.append(recommendation_paragraph)

            # Additional Information Section
            additional_info = """
            <b>Additional Information:</b><br/>
            Pneumonia is a serious condition that affects the lungs and can be life-threatening if not treated promptly.<br/>
            Early detection and proper treatment are key to improving recovery outcomes.<br/><br/>
            <b>Important Notes:</b><br/>
            - The prediction provided is based on a machine learning model and is not a definitive medical diagnosis.<br/>
            - The report should be used in conjunction with clinical judgment from a qualified healthcare provider.<br/>
            """
            additional_info_paragraph = Paragraph(additional_info, prediction_style)
            elements.append(additional_info_paragraph)

            # Add a hyperlink to book an appointment if pneumonia is detected
            if result[0][0] < 0.5:
                appointment_link = """
                <b>Book an Appointment:</b><br/>
                Please <a href="https://medipulse-parth-tyagi.netlify.app/">click here</a> to book an appointment with a doctor.
                """
                appointment_paragraph = Paragraph(appointment_link, prediction_style)
                elements.append(appointment_paragraph)

            # Finalize the document
            doc.build(elements)

            QMessageBox.information(None, "Success", "Report saved successfully!")



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())