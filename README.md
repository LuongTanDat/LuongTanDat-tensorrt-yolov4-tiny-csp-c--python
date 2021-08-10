# tensorrt-yolov4-tiny-csp-c--python

- Yolov4 Darknet

```
[v] cmake -D SINGLE_CAM=ON -D VIDEO_EXAMPLES=ON ..
[v] cmake -D SINGLE_CAM=ON -D CAM_ID_EXAMPLES=ON ..
[v] cmake -D FOUR_CAMS=ON -D VIDEO_EXAMPLES=ON ..
[v] cmake -D FOUR_CAMS=ON -D CAM_ID_EXAMPLES=ON ..
```
```
[v] cmake -D INFERENCE_VIDEO=ON -D VIDEO_EXAMPLES=ON ..
[v] cmake -D INFERENCE_VIDEO=ON -D CAM_ID_EXAMPLES=ON ..
[v] cmake -D INFERENCE_VIDEO=ON -D VIDEO_EXAMPLES=ON -D DEBUG=ON ..
[v] cmake -D INFERENCE_VIDEO=ON -D CAM_ID_EXAMPLES=ON -D DEBUG=ON ..
```
```
[v] cmake -D TENSORRT_API=ON ..

./Nobi_Trt \
    --engine-file "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.engine" \
    --label-file "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.names" \
    --dims 512 512 --obj-thres 0.3 --nms-thres 0.3 --type-yolo csp --dont-show
```
```
[v] cmake -D NOBI_CAMERA_AI_API=ON -D VIDEO_EXAMPLES=ON ..
[v] cmake -D NOBI_CAMERA_AI_API=ON -D CAM_ID_EXAMPLES=ON ..
[v] cmake -D NOBI_CAMERA_AI_API=ON -D VIDEO_EXAMPLES=ON -D DEBUG=ON ..
[v] cmake -D NOBI_CAMERA_AI_API=ON -D CAM_ID_EXAMPLES=ON -D DEBUG=ON ..

./Nobi_Camera_AI \
    --engine-file "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.engine" \
    --label-file "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v2/scaled_nobi_pose_v2.names" \
    --save-dir "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/AlphaPose/nobi-hw-videocapture/EMoi///" \
    --dims 512 512 --obj-thres 0.3 --nms-thres 0.3 --type-yolo csp --dont-show
```
```
[v] cmake -DINFERENCE_DARKNET=ON -DINFERENCE_VIDEO=ON -DVIDEO_EXAMPLES=ON ..
[v] cmake -DINFERENCE_DARKNET=ON -DTENSORRT_API=ON -DVIDEO_EXAMPLES=ON -DDEBUG=ON ..
./Nobi_App \
    --weights-file /mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/yolov4-csp/yolov4-csp-512.weights \
    --cfg-file /mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/yolov4-csp/yolov4-csp-512.cfg \
    --names-file /mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/yolov4-csp/yolov4-csp-512.names \
    --thresh 0.5 --dont-show
```