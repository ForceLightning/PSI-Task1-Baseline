# DCPose

# Requirements
Directory format:
```
.
├── datasets                    
├── demo                        code for video.py as well as input and output video
├── docs
├── DCPose_SUPP_DIR            Pretrained models and supplementary                      
├── engine                      
├── object detector             Pytorch models
|   └── YOLOv3                  Yolov3 Models and configurations
├── posetimation                
├── thirdparty                   
|   └── deform_conv              DCN                             
├── tools                   
├── utils                    
├── visualization                Code for Bounding box and Skeleton visualization             
├── .gitignore  
├── README.md                   
└── DCPose_requirements.txt      Dependencies in DCPose_requirements.txt form
```

## Development Environment
1. [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive) with an appropriately up-to-date Nvidia GPU.
2. [cuDNN v8.9.7](https://developer.nvidia.com/rdp/cudnn-archive) download the corresponding cuDNN version according to the CUDA Tooklit version installed
3. Python 3.6.12

# Installation for DCPose
1. Install python module dependencies with any of the following:
- Pip
```bash
pip install -r DCPOSE_requirements.txt
```
- Pipenv with `DCPOSE_requirements.txt`
```bash
pipenv install -r DCPOSE_requirements.txt
```
- Pipenv with `Pipfile`
```bash
pipenv install
```
2. Upgrade the setuptools wheels
- Pip
```bash
pip install -r --upgrade setuptools wheel
```

3. Install DCN
```bash
cd thirdparty/deform_conv
python setup.py develop
```

4. Download the [pretrained models](https://drive.google.com/drive/folders/1VPcwo9jVhnJpf5GVwR2-za1PFE6bCOXE) and put in the directory DcPose_supp_files

Note that if a different CUDA Toolkit version is installed, modify the source for PyTorch wheels based on the CUDA Toolkit version.

# Usage
## Run video.py
1. Create an input directory to put our videos 
```bash
cd demo/
mkdir input/
```
2. Run `video.py`
```bash
python video.py
```
