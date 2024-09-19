#rm -rf  /project/train/src_repo/trainval/*
#rm -rf /project/train/src_repo/*.txt
#rm -rf /project/train/src_repo/*.cache
cp -r /home/data/2978 /project/train/src_repo/trainval
cp -r /home/data/2979 /project/train/src_repo/trainval
python /project/train/src_repo/split.py
cd /project/train/src_repo/ultralytics/
yolo train model=/project/train/src_repo/ultralytics/ultralytics/cfg/models/v8/yolov8s-p2.yaml  data=/project/train/src_repo/ultralytics/car.yaml project=/project/train/models/   batch=32 epochs=200  device=0   close_mosaic=20
#yolo train model=/project/train/src_repo/ultralytics/yolov8s.pt   data=/project/train/src_repo/ultralytics/car.yaml project=/project/train/models/   batch=80 epochs=300  device=0  scale=0  mosaic=0.9 close_mosaic=40
#yolo train model=/project/train/src_repo/ultralytics/yolov8m.pt   data=/project/train/src_repo/ultralytics/car.yaml project=/project/train/models/   batch=48 epochs=200  device=0  scale=0  mosaic=0.9 close_mosaic=40
