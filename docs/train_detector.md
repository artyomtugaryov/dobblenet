# How to train YoloV4 Dobblenet in the Darknet format

0. Clone the Dobblenet repository:
    ```sh
    git clone git@github.com:artyomtugaryov/dobblenet.git
    cd dobblenet
    ```

1. Create a virtual environment to work with python:
    ```sh
    python3 -m pip install virtualenv
    python3 -m virtualenv venv
    source venv/bin/activate
    
    python3 -m pip install -r requirements.txt
    ```

2. Download the Dobble dataset (dobblenet_dataset.zip) from [GitHub releases](https://github.com/artyomtugaryov/dobblenet/releases/latest):
    ```sh
    python3 scripts/download_dataset.py --dataset-link https://github.com/artyomtugaryov/dobblenet/releases/download/alpha0.2/dobblenet_dataset.zip
    ```

    The dataset will apear in the `dataset` folder in the root of the repository. 

3. Clone the Darknet repository from the [AlexeyAB fork](https://github.com/AlexeyAB/darknet):
    ```sh
    git clone git@github.com:AlexeyAB/darknet.git
    ```

4. Move dataset files to the folder with the darknet repository:
    ```sh
    python3 scripts/spread_dataset.py
    ```

5. Open the `Makefile` inside the darknet repository and change:
    If you have set up GPU and CUDNN:
        1. `GPU=0` -> `GPU=1` 
        2. `CUDNN=0` -> `CUDNN=1`
    In addition it is to recomended to speed up training stage to change:
        1. `OPENCV=0` -> `OPENCV=1`
        2. `AVX=0` -> `AVX=1`

7. Initialize the CUDA environment:
    ```sh
    export LD_LIBRARY_PATH="/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH"
    export PATH="/usr/local/cuda-11.0/bin/:$PATH"
    ```

8. Compile darknet:
    ```sh
    cd darknet
    make -j 4
    ```

    Ater successfully build you can find `darknet` binary file in the root of the darknet repository 

9. Download the pre-trained YOLOv4 weights:
    ```sh
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
    ```

10. Copy the dobblenet config to `darknet/cfg` folder:
    ```sh
    ./darknet detector train data/obj.data ../configs/dobblenet.darknet.cfg yolov4.conv.137 -dont_show -map
    ```

    This process takes much time - around 24 hours (depends on GPU). The first output looks like the following:

    ```
    CUDA-version: 11000 (11060), cuDNN: 8.0.4, GPU count: 1  
    OpenCV version: 4.2.0
    Prepare additional network for mAP calculation...
    0 : compute_capability = 750, cudnn_half = 0, GPU: NVIDIA GeForce GTX 1650 with Max-Q Design 
    net.optimized_memory = 0 
    mini_batch = 1, batch = 64, time_steps = 1, train = 0 
    layer   filters  size/strd(dil)      input                output
    0 Create CUDA-stream - 0 
    Create cudnn-handle 0 
    conv     32       3 x 3/ 1    416 x 416 x   3 ->  416 x 416 x  32 0.299 BF
    ...
    ```

    You can stop the training through 24 - 48 hours.
    The results (weights) are in `darknet/trainig` folder - `_best.weights` and `_last.weights`
    