# ☆ Export paths: ☆

```bash
export WEIGHTS="/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v3/scaled_nobi_pose_v3.weights"
export NAMES="/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v3/scaled_nobi_pose_v3.names"
export CFG="/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v3/scaled_nobi_pose_v3.cfg"
export SAVE_DIR="/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/nobi-hw-videocapture/EMoi///"
export ENGINE="/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_model_v3/scaled_nobi_pose_v3.engine"
export ALPHAPOSE_MODEL="/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/AlphaPose/AlphaPose_TorchScript/model-zoo/fast_pose_res50/fast_res50_256x192.jit"
export TABULAR_MODEL="/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/model-zoo/nobi_tabular_v1/tabular_v1.jit"
```

- If you using `darknet` for detector:

```bash
cd darknet
make -j$(nproc)
```

# ☆ API like darknet ☆

<details>
  <summary>RUN WITHOUT JSON</summary>

## Build test object detection by TensorRT model on a specific image path like darknet

```bash
rm -rf .cmake/ *
cmake -DTENSORRT_API=ON ..
cmake --build . --config Release
./Nobi_App --engine-file ${ENGINE} --label-file ${NAMES} --dims 512 512 --obj-thres 0.3 --nms-thres 0.4 --type-yolo csp --dont-show
```

## Build test object detection by darknet C++ model on a specific image path like darknet

```bash
rm -rf .cmake/ *
cmake -DTENSORRT_API=ON -DINFERENCE_DARKNET=ON ..
cmake --build . --config Release
./Nobi_App --weights-file ${WEIGHTS} --cfg-file ${CFG} --names-file ${NAMES} --thresh 0.3 --dont-show
```

## Build test object detection by TensorRT model on a specific image path like darknet, and support **_pose estimation_**

```bash
rm -rf .cmake/ *
cmake -DTENSORRT_API=ON -DINFERENCE_ALPHAPOSE_TORCH=ON ..
cmake --build . --config Release
./Nobi_App --engine-file ${ENGINE} --label-file ${NAMES} --alphapose-jit ${ALPHAPOSE_MODEL} --dims 512 512 --obj-thres 0.3 --nms-thres 0.4 --type-yolo csp --dont-show
```

## Build test object detection by darknet C++ model on a specific image path like darknet, and support **_pose estimation_**

```bash
rm -rf .cmake/ *
cmake -DTENSORRT_API=ON -DINFERENCE_DARKNET=ON -DINFERENCE_ALPHAPOSE_TORCH=ON ..
cmake --build . --config Release
./Nobi_App --weights-file ${WEIGHTS} --cfg-file ${CFG} --names-file ${NAMES} --alphapose-jit ${ALPHAPOSE_MODEL} --thresh 0.3 --dont-show
```

</details>

<details>
  <summary>RUN WITH JSON</summary>

## Build test object detection by TensorRT model on a sample video path like darknet and output `Json`

```bash
rm -rf .cmake/ *
cmake -DTENSORRT_API=ON -DJSON=ON ..
cmake --build . --config Release
./Nobi_App --engine-file ${ENGINE} --label-file ${NAMES} --dims 512 512 --obj-thres 0.3 --nms-thres 0.4 --type-yolo csp --dont-show
```

## Build test object detection by darknet C++ model on a sample video path like darknet and output `Json`

```bash
rm -rf .cmake/ *
cmake -DTENSORRT_API=ON -DINFERENCE_DARKNET=ON -DJSON=ON ..
cmake --build . --config Release
./Nobi_App --weights-file ${WEIGHTS} --cfg-file ${CFG} --names-file ${NAMES} --thresh 0.3 --dont-show
```

## Build test object detection by TensorRT model on a sample video path like darknet, and support **_pose estimation_** and output `Json`

```bash
rm -rf .cmake/ *
cmake -DTENSORRT_API=ON -DINFERENCE_ALPHAPOSE_TORCH=ON -DJSON=ON ..
cmake --build . --config Release
./Nobi_App --engine-file ${ENGINE} --label-file ${NAMES} --alphapose-jit ${ALPHAPOSE_MODEL} --dims 512 512 --obj-thres 0.3 --nms-thres 0.4 --type-yolo csp --dont-show
```

## Build test object detection by TensorRT model on a sample video path like darknet, and support **_pose estimation_**, **_tabular learner_** and output `Json`

```bash
rm -rf .cmake/ *
cmake -DTENSORRT_API=ON -DINFERENCE_ALPHAPOSE_TORCH=ON -DINFERENCE_TABULAR_TORCH=ON -DJSON=ON ..
cmake --build . --config Release
./Nobi_App --engine-file ${ENGINE} --label-file ${NAMES} --alphapose-jit ${ALPHAPOSE_MODEL} --tabular-jit ${TABULAR_MODEL} --dims 512 512 --obj-thres 0.3 --nms-thres 0.4 --type-yolo csp --dont-show
```

## Build test object detection by darknet C++ model on a sample video path like darknet, and support **_pose estimation_** and output `Json`

```bash
rm -rf .cmake/ *
cmake -DTENSORRT_API=ON -DINFERENCE_DARKNET=ON -DINFERENCE_ALPHAPOSE_TORCH=ON -DJSON=ON ..
cmake --build . --config Release
./Nobi_App --weights-file ${WEIGHTS} --cfg-file ${CFG} --names-file ${NAMES} --alphapose-jit ${ALPHAPOSE_MODEL} --thresh 0.3 --dont-show
```

</details>
