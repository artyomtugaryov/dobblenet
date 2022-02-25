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

6. Compile darknet:
    ```sh
    make -j 4
    ```