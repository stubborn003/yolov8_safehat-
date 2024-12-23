import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from datetime import datetime
from ultralytics import YOLO

CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask",
    "NO-Safety Vest", "Person", "Safety Cone",
    "Safety Vest", "machinery", "vehicle"
]


class MWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUI()
        self.model = YOLO('best.pt')  # 确保路径正确
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.timeout.connect(self.show_camera)

    def setupUI(self):
        self.setWindowTitle('头盔检测')
        self.setMinimumSize(1200, 800)

        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)
        mainLayout = QtWidgets.QVBoxLayout(centralWidget)

        self.label_ori_video = QtWidgets.QLabel()
        self.label_treated = QtWidgets.QLabel()
        self.label_ori_video.setMinimumSize(640, 480)
        self.label_treated.setMinimumSize(640, 480)

        videoLayout = QtWidgets.QHBoxLayout()
        videoLayout.addWidget(self.label_ori_video)
        videoLayout.addWidget(self.label_treated)
        mainLayout.addLayout(videoLayout)

        groupBox = QtWidgets.QGroupBox("控制面板", self)
        controlLayout = QtWidgets.QVBoxLayout(groupBox)
        self.textLog = QtWidgets.QTextBrowser()
        controlLayout.addWidget(self.textLog)

        btnLayout = QtWidgets.QHBoxLayout()
        self.videoBtn = QtWidgets.QPushButton('视频文件')
        self.camBtn = QtWidgets.QPushButton('摄像头')
        self.stopBtn = QtWidgets.QPushButton('🛑停止')

        for button in [self.videoBtn, self.camBtn, self.stopBtn]:
            button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; } "
                                 "QPushButton:hover { background-color: #45a049; }")
        btnLayout.addWidget(self.videoBtn)
        btnLayout.addWidget(self.camBtn)
        btnLayout.addWidget(self.stopBtn)
        controlLayout.addLayout(btnLayout)
        mainLayout.addWidget(groupBox)
        mainLayout.setStretch(0, 1)

        self.videoBtn.clicked.connect(self.startVideo)
        self.camBtn.clicked.connect(self.startCamera)
        self.stopBtn.clicked.connect(self.stop)

    def log(self, message):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{current_time}] {message}\n"
        self.textLog.insertPlainText(log_message)
        self.textLog.moveCursor(QtGui.QTextCursor.Start)

    def startCamera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.log("无法打开摄像头")
            return
        self.log("摄像头已启动")
        self.timer_camera.start(30)  # 设置间隔以适应帧率

    def startVideo(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择视频文件", "",
                                                            "所有文件 (*);;视频文件 (*.mp4 *.avi)", options=options)
        if fileName:
            self.cap = cv2.VideoCapture(fileName)
            if not self.cap.isOpened():
                self.log("无法打开视频文件")
                return
            self.log("视频文件已启动")
            self.timer_camera.start(30)  # 设置间隔以适应帧率

    def show_camera(self):
        ret, frame = self.cap.read()
        if not ret:
            self.log("无法读取帧")
            return

            # 指定统一的帧大小，例如640x480
        target_size = (640, 480)
        frame_resized = cv2.resize(frame, target_size)

        # 显示原视频帧
        self.display_frame(frame_resized)
        # 处理并绘制检测框
        self.detect_objects(frame_resized)

    def detect_objects(self, frame):
        results = self.model(frame)  # 传递调整后的框
        detected_boxes = []
        hardhat_count = 0
        no_hardhat_count = 0

        for result in results[0].boxes:
            class_id = int(result.cls[0])
            confidence = float(result.conf[0])
            if class_id in [0, 2]:  # 仅处理"Hardhat"和"NO-Hardhat"
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                detected_boxes.append({
                    'bbox': (x1, y1, x2, y2),
                    'class_name': CLASS_NAMES[class_id],
                    'confidence': confidence
                })

                # 计数 "Hardhat" 和 "NO-Hardhat"
                if class_id == 0:
                    hardhat_count += 1
                elif class_id == 2:
                    no_hardhat_count += 1

                    # 更新日志，打印检测到的数量和时间
        if hardhat_count > 0 or no_hardhat_count > 0:
            # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log(f"检测到 Hardhat: {hardhat_count}, NO-Hardhat: {no_hardhat_count}")

            # 在处理后视频帧上绘制检测框
        for box in detected_boxes:
            x1, y1, x2, y2 = box['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # 更新处理后帧显示
        self.update_treated_frame(frame)

    def update_treated_frame(self, frame):
        qImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0],
                              QtGui.QImage.Format_BGR888)
        self.label_treated.setPixmap(QtGui.QPixmap.fromImage(qImage))

    def display_frame(self, frame):
        qImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0],
                              QtGui.QImage.Format_BGR888)
        self.label_ori_video.setPixmap(QtGui.QPixmap.fromImage(qImage))

    def stop(self):
        self.timer_camera.stop()
        if hasattr(self, 'cap'):
            self.cap.release()
            self.log("摄像头或视频文件已停止")
        self.label_ori_video.clear()
        self.label_treated.clear()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MWindow()
    window.show()
    sys.exit(app.exec_())