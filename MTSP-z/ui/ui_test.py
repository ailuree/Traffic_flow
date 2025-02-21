from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from custom_grips import CustomGrip  # 确保正确导入你之前定义的 CustomGrip 类

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Custom Borderless Resizable Window")
        self.setGeometry(100, 100, 800, 600)

        # 创建一个自定义的窗口
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        # 启用无边框窗口
        self.setWindowFlags(Qt.FramelessWindowHint)

        # 创建自定义大小调整器并添加到不同位置
        self.top_grip = CustomGrip(self, Qt.TopEdge)
        self.bottom_grip = CustomGrip(self, Qt.BottomEdge)
        self.left_grip = CustomGrip(self, Qt.LeftEdge)
        self.right_grip = CustomGrip(self, Qt.RightEdge)

        # 设置主窗口布局
        layout = QVBoxLayout(self.main_widget)
        self.main_widget.setLayout(layout)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
