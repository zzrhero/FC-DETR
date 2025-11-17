import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

# onnx onnxsim onnxruntime onnxruntime-gpu



if __name__ == '__main__':
    model = RTDETR('rtdetr-weight-path')
    model.export(format='onnx', simplify=True)