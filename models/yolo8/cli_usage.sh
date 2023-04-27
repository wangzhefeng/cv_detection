# ------------------------------------------------------------
# 语法
# yolo TASK MODE ARGS
# TASK: [detect, setment, classify]
# MODE: [train, val, predict, export, track]
# ARGS: https://docs.ultralytics.com/usage/cfg/ or $ yolo cfg
# ------------------------------------------------------------
# train
yolo trian data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01

# predict
yolo predict model=yolov8n-seg.pt source='https://youtu.be/Zgi9g1ksQHc' imgsz=320

# val
yolo val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640

# export
yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128

# sepcial
yolo help
yolo checks
yolo version
yolo settings
yolo copy-cfg
yolo cfg
