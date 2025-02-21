## 核心模块说明

1. **主程序模块**
   - main.py: 程序入口，包含主窗口逻辑
   - app.py: Web服务入口，提供Web访问接口

2. **检测核心模块**
   - classes/yolo.py: YOLO检测实现
   - classes/paint_trail.py: 轨迹绘制实现
   - classes/car_chart.py: 流量统计图实现

3. **UI模块**
   - ui/main_window.py: 主界面UI
   - ui/dialog/*: 各种对话框UI
   - ui/pop & ui/toast: 提示组件

4. **工具模块**
   - utils/main_utils.py: 通用工具函数
   - utils/AtestCamera.py: 摄像头工具

5. **配置模块**
   - config/config.json: 配置文件
   - classes/main_config.py: 配置管理类

6. **数据存储**
   - models/: 模型文件存储
   - pre_result/: 检测结果存储
   - pre_labels/: 检测标签存储

# PyTorch 模型加载警告说明

## 警告内容解析

```python
FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value)
```

这表示当前使用的是默认的模型加载方式，可能存在安全隐患。

### 主要问题
1. **安全风险**：
   - 当前的 `torch.load()` 使用 Python 的 pickle 模块
   - pickle 在加载时可能执行任意代码
   - 如果加载恶意构造的模型文件，可能导致安全问题

2. **未来变更**：
   - 在未来版本中，`weights_only` 默认值将改为 `True`
   - 这将限制反序列化过程中可执行的函数
   - 只允许加载模型权重，而不是完整的 Python 对象

### 解决方案

1. **安全加载方式**：
```python
# yolo.py 中修改模型加载
def load_yolo_model(self):
    if self.used_model_name != self.new_model_name:
        # 使用安全的加载方式
        model = YOLO(self.new_model_name, weights_only=True)
        self.used_model_name = self.new_model_name
        return model
```

2. **添加安全白名单**：
```python
from torch.serialization import add_safe_globals

# 添加可信任的全局对象
add_safe_globals({
    'your_trusted_module': your_trusted_module
})
```

3. **使用验证过的模型**：
- 只使用来源可信的模型文件
- 验证模型文件的完整性
- 避免加载未知来源的模型

### 其他信息说明
```
YOLOv8n summary (fused): 168 layers, 3151904 parameters, 0 gradients, 8.7 GFLOPs
```
这部分显示了模型的基本信息：
- 使用的是 YOLOv8n 模型
- 包含 168 层
- 有约 315 万个参数
- 计算量为 8.7 GFLOPs
