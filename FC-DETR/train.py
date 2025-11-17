import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-r50-FC-DETR.yaml')
    # model.load('') #
    model.train(data='dataset/visdrone.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=2,
                workers=4,
                # device='0,1', #
                # resume='', # last.pt path
                project='runs/train',
                name='exp',
                )