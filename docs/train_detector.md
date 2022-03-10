# How to train YoloV4 Dobblenet in the Darknet format

Once upon a time, when everyone worked from the office and did not think about remote work, my colleagues and I decided to take a break from work and play game that can distract us from work. We chose to play the Dobble (Spot it). Dobble is a speed and observation game for the everyone. The aim of the game -
Each two cards have one symbol in common. You need to be the first to find and name it to win the card.

After a number of games played we began to talk about what would be nice to create a programm that can help you to play Dobble or for example could be an opponent for you. This is a speed game, so a computer can be more faster than a the most speedy human. So the idea of creating an application for playing Dobble was born at that moment.

We are working in a big OpenVINO team that are creating the framework for optimization and inference neural networks. Of course we wanted to use our framework to built our tools and a neural network should be the cornerstone of the tool. 

It was absolutelly clear for us that the neural network shoud solve Object Detection task and the most popular model to solve OD task was YoloV4, we decided to use this topology.

We understood that the we need a lot of data to train our custom YoloV4. We started to collect the dataset from taking photos of the dobble cards that we had. The decl of cards of this game collects 55 and we took around 4 photos of each card in different angles. There are around 390 photos as results. 
But taking the photos is not the main problem of dataset collection, the main challenge is annotate each icon on each photo. There are 8 images in each photos, so there are 3120 in the 390 images. Annotate means dedicate cooddinats of each object in each photo. There are many different dataset formats that dedicates formats of storing images and annotations for each image. 
To annotate the dataset we used CVAT tool that provides usefull interface for annoations: in each image you need to draw a rectangle around each object as it is shown in the picture:
![](./images/cvat.jpg) 

This is a hard work that was done for a week or two, and you can find the results of this wor in the kaggle: https://www.kaggle.com/atugaryov/dobble
But 390 images is not enough to train a model. To get more images we used roboflow tool to augment images and have more data - around 900 images. The dataset was splited to 3 subset: train, validate and test. This is nessesary for successfully training. The result dataset you can find in the releases of the reposiroy.

The data was ready for train and we started to search info about transfer learning of YoloV4 network. There are many materials about this misteral process. 
One of the most important part of training is hardware with GPU. We were lucky - we had a laptop with Nvidia GPU. An we have to setup environment for GPU using (set up drivers and CUDA environment). THe full instructions you can find in the Nvidia materials.

The next step is preparing environment for transfer learning of the YoloV4. We used Darkbent repository for it:

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
    