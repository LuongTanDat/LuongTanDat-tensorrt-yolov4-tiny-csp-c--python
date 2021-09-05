FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
ENV DEBIAN_FRONTEND noninteractive

# Install git và python==3.6
RUN apt update && apt install --assume-yes software-properties-common
# RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update && apt install --assume-yes python3.6 python3.6-dev python3-pip
RUN ln -sfn /usr/bin/python3.6 /usr/bin/python3
RUN ln -sfn /usr/bin/python3 /usr/bin/python
RUN ln -sfn /usr/bin/pip3 /usr/bin/pip
RUN apt install --assume-yes git tmux tree

# install boost
COPY ./docker-resource/boost_1_77_0.tar.gz /root/boost_1_77_0.tar.gz
WORKDIR /root
RUN apt install --assume-yes build-essential g++ python-dev autotools-dev libicu-dev build-essential libbz2-dev libboost-all-dev
RUN tar -xvzf boost_1_77_0.tar.gz
RUN cd boost_1_77_0 &&\
    ./bootstrap.sh --prefix=/usr/ --with-libraries=python &&\
    ./b2 --with=all -j$(nproc) install

# install Gstreamer
WORKDIR /root
RUN apt update
RUN apt upgrade --assume-yes
RUN apt install --assume-yes libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio

# install nvtop htop
WORKDIR /root
RUN apt install cmake libncurses5-dev libncursesw5-dev git -yq
RUN git clone https://github.com/Syllo/nvtop.git
RUN mkdir -p nvtop/build && cd nvtop/build &&\
    cmake .. &&\
    make -j$(nproc) &&\
    make install
RUN apt install htop

# install TensorRT
COPY ./docker-resource/TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1.tar.gz /root/TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1.tar.gz
WORKDIR /root
RUN tar -xvzf /root/TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1.tar.gz -C /usr/local/
WORKDIR /usr/local
RUN ln -s TensorRT-7.2.3.4 TensorRT

# install opencv
RUN apt update
RUN apt upgrade -y
RUN apt install build-essential cmake pkg-config unzip yasm git checkinstall libjpeg-dev libpng-dev libtiff-dev  libavcodec-dev libavformat-dev libswscale-dev libavresample-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev  libfaac-dev libmp3lame-dev libvorbis-dev libopencore-amrnb-dev libopencore-amrwb-dev libdc1394-22 libdc1394-22-dev libxine2-dev libv4l-dev v4l-utils libgtk-3-dev python3-dev python3-pip libtbb-dev libatlas-base-dev gfortran libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev libgphoto2-dev libeigen3-dev libhdf5-dev doxygen -y
RUN pip3 install -U pip numpy
WORKDIR /usr/include/linux
RUN ln -s -f ../libv4l1-videodev.h videodev.h
WORKDIR /root
RUN export HOME=$(pwd)
RUN git clone https://github.com/opencv/opencv.git
RUN git clone https://github.com/opencv/opencv_contrib.git
RUN mkdir -p opencv/build
WORKDIR /root/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_C_COMPILER=/usr/bin/gcc \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=OFF \
    -D WITH_TBB=ON \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D BUILD_opencv_cudacodec=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D WITH_V4L=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENGL=ON \
    -D WITH_GSTREAMER=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_PC_FILE_NAME=opencv.pc \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D OPENCV_EXTRA_MODULES_PATH=$HOME/opencv_contrib/modules \
    -D BUILD_SHARED_LIBS=ON \
    -D WITH_FFMPEG=OFF \
    -D WITH_OPENCL=ON \
    -D BUILD_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D PYTHON_EXECUTABLE=/usr/bin/python3 \
    -D OPENCV_PYTHON3_INSTALL_PATH=/usr/local/lib/python3.6/dist-packages/ \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF ..
RUN make -j$(nproc)
RUN make install
RUN ldconfig
RUN pkg-config --modversion opencv

# install pytorch
WORKDIR /root
RUN pip3 install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
RUN git clone --recursive https://github.com/pytorch/pytorch
WORKDIR /root/pytorch
RUN export CMAKE_PREFIX_PATH=/usr
RUN python3 setup.py install

# install curlpp
WORKDIR /root
RUN apt install libcurl4-openssl-dev
RUN git clone https://github.com/jpbarrette/curlpp.git
RUN mkdir -p curlpp/build && cd curlpp/build
WORKDIR /root/curlpp/build
RUN cmake ..
RUN make -j$(nproc)
RUN make install

RUN echo 'export PATH="/usr/local/cuda/bin${PATH:+:${PATH}}"' >>/root/.bashrc
RUN echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/TensorRT/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"' >>/root/.bashrc
RUN echo "export Torch_DIR=/usr/local/lib/python3.6/dist-packages/torch" >>/root/.bashrc

WORKDIR /root
RUN git clone https://github.com/LuongTanDat/yolov4-v4csp-tensorrt-darknet-cpp-python.git -b docker-nobi-API nobi-hw-videocapture
RUN mkdir -p model-zoo

WORKDIR /root/nobi-hw-videocapture/darknet
RUN make -j$(nproc)
RUN ln -sf $(readlink -f include/yolo_v2_class.hpp) /usr/local/include/yolo_v2_class.hpp
RUN ln -sf $(readlink -f libdarknet.so) /usr/local/lib/libdarknet.so

WORKDIR /root/nobi-hw-videocapture/
RUN mkdir -p build_tensorrt
RUN mkdir -p build_darknet
RUN mkdir -p build_tensorrt_pose
RUN mkdir -p build_tensorrt_pose_tabular
RUN mkdir -p build_darknet_pose

WORKDIR /root/nobi-hw-videocapture/build_tensorrt
RUN export PATH="/usr/local/cuda/bin${PATH:+:${PATH}}" && \
    export Torch_DIR="/usr/local/lib/python3.6/dist-packages/torch" && \
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/TensorRT/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" && \
    cmake -DJSON=ON ..
RUN cmake --build . --config Release
# WORKDIR /root/nobi-hw-videocapture/build_darknet
# RUN ln -sf $(readlink -f ../darknet/include/yolo_v2_class.hpp) /usr/local/include/yolo_v2_class.hpp && \
#     ln -sf $(readlink -f ../darknet/libdarknet.so) /usr/local/lib/libdarknet.so && \
#     export PATH="/usr/local/cuda/bin${PATH:+:${PATH}}" && \
#     export Torch_DIR="/usr/local/lib/python3.6/dist-packages/torch" && \
#     export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/TensorRT/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" && \
#     cmake -DINFERENCE_DARKNET=ON -DJSON=ON ..
# RUN cmake --build . --config Release
WORKDIR /root/nobi-hw-videocapture/build_tensorrt_pose
RUN export PATH="/usr/local/cuda/bin${PATH:+:${PATH}}" && \
    export Torch_DIR="/usr/local/lib/python3.6/dist-packages/torch" && \
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/TensorRT/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" && \
    cmake -DINFERENCE_ALPHAPOSE_TORCH=ON -DJSON=ON ..
RUN cmake --build . --config Release
WORKDIR /root/nobi-hw-videocapture/build_tensorrt_pose_tabular
RUN export PATH="/usr/local/cuda/bin${PATH:+:${PATH}}" && \
    export Torch_DIR="/usr/local/lib/python3.6/dist-packages/torch" && \
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/TensorRT/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" && \
    cmake -DINFERENCE_ALPHAPOSE_TORCH=ON -DINFERENCE_TABULAR_TORCH=ON -DJSON=ON ..
RUN cmake --build . --config Release
# WORKDIR /root/nobi-hw-videocapture/build_darknet_pose
# RUN ln -sf $(readlink -f ../darknet/include/yolo_v2_class.hpp) /usr/local/include/yolo_v2_class.hpp && \
#     ln -sf $(readlink -f ../darknet/libdarknet.so) /usr/local/lib/libdarknet.so && \
#     export PATH="/usr/local/cuda/bin${PATH:+:${PATH}}" && \
#     export Torch_DIR="/usr/local/lib/python3.6/dist-packages/torch" && \
#     export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/TensorRT/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" && \
#     cmake -DINFERENCE_DARKNET=ON -DINFERENCE_ALPHAPOSE_TORCH=ON -DJSON=ON ..
# RUN cmake --build . --config Release

# RUN rm -rf /root/TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1.tar.gz
# RUN rm -rf /root/boost_1_77_0
# RUN rm -rf /root/boost_1_77_0.tar.gz
# RUN rm -rf /root/nvtop
# RUN rm -rf /root/pytorch
# RUN rm -rf /root/curlpp

WORKDIR /root/nobi-hw-videocapture/
EXPOSE 2210
WORKDIR /root/nobi-hw-videocapture/build_tensorrt_pose
CMD ["./Nobi_App", "--engine-file", "/root/model-zoo/nobi.engine", "--label-file", "/root/model-zoo/nobi.names", "--alphapose-jit", "/root/model-zoo/pose.jit", "--port", "2210", "--dims", "512", "512", "--obj-thres", "0.3", "--nms-thres", "0.4", "--type-yolo", "csp", "--dont-show"]

# docker build --tag hienanh/nobi-api:0.2.3 .

# export ENGINE=/mnt/4B323B9107F693E2/TensorRT/model-zoo/nobi_model_v3/scaled_nobi_pose_v3.engine
# export NAMES=/mnt/4B323B9107F693E2/TensorRT/model-zoo/nobi_model_v3/scaled_nobi_pose_v3.names
# export ALPHAPOSE_MODEL=/mnt/4B323B9107F693E2/TensorRT/model-zoo/fast_pose_res50/fast_res50_256x192.jit
# docker run --rm --gpus all -p 2210:2210 \
# -v ${ENGINE}:/root/model-zoo/nobi.engine \
# -v ${NAMES}:/root/model-zoo/nobi.names \
# -v ${ALPHAPOSE_MODEL}:/root/model-zoo/pose.jit \
# hienanh/nobi-api:0.2.3
