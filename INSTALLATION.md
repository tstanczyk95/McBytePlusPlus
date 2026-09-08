# McByte++ installation

The installation is pretty straightforward. For the smooth experience, please follow the exact order of the instructions.


## Versioning

This installation has been tested and works well on Linux with <b>CUDA 12.8</b> and <b>GCC 10.5</b>.<br/>
Please have these two installed and enabled/loaded first.<br/>
We also recommend using the same versions of the other packages and libraries as specified below.


## Virtual environment

Install [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install). Based on your needs and resources (e.g. remote server), consider [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) instead.


Create and activate a conda environment with Python as specified:
```
conda create -n mcbyteplusplus python=3.10
conda activate mcbyteplusplus
```


## Main tracker installation

Clone the repository into your folder and go to McByte++ root:
```
git clone https://github.com/tstanczyk95/McBytePlusPlus.git
cd McBytePlusPlus
```

Install PyTorch and related packages as specified below. Command taken from [here](https://pytorch.org/get-started/previous-versions/). 
```
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

Install the required packages following the instructions:
```
pip3 install -r requirements.txt
pip3 install -e . --no-build-isolation
pip3 install cython
python3 -m pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' --no-build-isolation
pip3 install cython_bbox
pip3 install --upgrade numpy==1.24.4
```
<i>(In case see an error in red about pip's dependency resolver, do not worry about it, you will be able to proceed and still finish the installation and run Mcbyte++)</i>

Make directory for the pretrained detector ([YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), already installed) models:
```
mkdir pretrained
```

Download the following pre-trained models from their original sources:
- [SportsMOT](https://github.com/MCG-NJU/MixSort?tab=readme-ov-file#model-zoo): yolox_x_sports_mix.pth.tar 🔥 <b> Recommended for the demo in your sport setting frames </b> 🔥
- [DanceTrack](https://huggingface.co/noahcao/dancetrack_models/tree/main/bytetrack_models): bytetrack_model.pth.tar


Place them in the <i>pretrained</i> folder.<br/>
If you cannot reach some of these models, then please see the last section of this page.


## Segmentation mask (creation and propagation) installation

Install the temporally propagated segmentation mask functionality ([EdgeTAM](https://github.com/facebookresearch/EdgeTAM)) as follows:
```
cd mask_propagation/EdgeTAM
pip3 install -e .
```

Download the model weights from the original EdgeTAM [repository](https://github.com/facebookresearch/EdgeTAM/tree/main/checkpoints/edgetam.pt). Place them in:
```
mask_propagation/EdgeTAM/checkpoints/
```
If you cannot reach this model, then please see the last section of this page.

## Person re-ID (optional, yet encouraged)

Install the [re-ID (torchreid)](https://github.com/kaiyangzhou/deep-person-reid) for tracklet re-ID joining:

```
cd ../../deep-person-reid
pip3 install -r requirements.txt
pip3 install "setuptools==59.8.0"
python3 setup.py develop
```

Download the model weights from the original GTA-link [repository](https://github.com/sjc042/gta-link/blob/main/reid_checkpoints/sports_model.pth.tar-60). Place them in:
```
mkdir pretrained/
```
If you cannot reach this model, then please see the last section of this page.


## Error handling
### Numpy still as >= 2.0
If running McByte gives you an error about Numpy 2.0 or higher, run this line one more time:
```
 pip3q install --upgrade numpy==1.24.4
```
It might happen that EdgeTAM installation updates this version by default, after running this line for the first time above, hence you need to run it again. (The same remark with the error in red applies here.)
### Unrecognized function arguments
If you modify function signatures or pull an updated version, and it will result in an error of unrecognized arguments, please run the setup install instruction again in the main McByte folder:
```
pip3 install -e . --no-build-isolation
```
It will adjust the running settings to the new signatures, etc.

## Pretrained model backups.

In case the original models links are broken and they cannot be downloaded, you can also reached them [here](https://drive.google.com/drive/folders/1yzzJk9dpJUY3lIHdkkFyGtKL2F-FenN6?usp=sharing). The following models are included: <br/>
- YOLOX SportsMOT: yolox_x_sports_mix.pth.tar
- YOLOX DanceTrack: bytetrack_x_dancetrack.pth.tar
- EdgeTAM: edgetam.pt
- Re-ID (GTA-link): sports_model.pth.tar-60
<br/>

(Credit belongs to the original authors, see the references in the sections above.)
