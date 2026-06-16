# TSegAgent: Zero-Shot Tooth Segmentation via Geometry-Aware Vision-Language Agents

This is the official implementation of MICCAI 2026 paper "TSegAgent: Zero-Shot Tooth Segmentation via Geometry-Aware Vision-Language Agents".
---

## Installation

This project is based on [`SAM3`](https://github.com/facebookresearch/sam3) and any VLM services. Make sure you have the right access to [`SAM3`](https://github.com/facebookresearch/sam3) weights.

### Prerequisites

- Python 3.12 or higher
- PyTorch 2.7 or higher
- CUDA-compatible GPU with CUDA 12.6 or higher
- C++ 17 or higher
- OpenGL dev libraries

#### 1. Clone the repository and update the dependencies

```shell
git submodule update --init --recursive
```

#### 2. Install python dependencies

```shell
pip install -r requirements.txt
```

#### 3. Install SAM3

You have to follow the full instructions in the `SAM3` repository [https://github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3).

Or else, you can simply run the following commands.

```shell
cd dep/sam3
pip install -e .
```

#### 4. Install `reversible_rasterizer`

This repository is for mesh rendering and pixel-face relation building.

```shell
cd dep/reversible_rasterizer
python setup.py install
```

#### 5. Prepare the `.env` file

Put your `OPENAI_API_KEY` or `ARK_API_KEY` in the `.env` file.

The VLM tooth classifier currently supports `chatgpt-5.2` and `doubao-seed 1.8` models. These models performanced well in our experiments.

```
OPENAI_API_KEY=sk-xxxx

ARK_API_KEY=xxxx
```


## Dataset

The project is tested on `Teeth3DS` dataset, which can be downloaded from [https://osf.io/xctdy/overview](https://osf.io/xctdy/overview).

The dataset is then processed with the following structure:

```
teeth3ds-dataset-root/
├─ Teeth3DS_train_test_split/
│  ├─ testing-upper.txt
│  ├─ testing-lower.txt
├─ obj/
│  ├─ 00OMSZGW/
│  │  ├─ 00OMSZGW_lower.obj
│  │  ├─ 00OMSZGW_lower.json

```

We put all the upper, lower jaw and GT labels inside one case folder for convinience. If you would like to test on other dataset, just keep the same structure.


## Test on Dataset

The `test_teeth3ds.py` is built for testing our code on `Teeth3DS` dataset.

Simply run:

```shell
python test_teeth3ds.py -d /path/to/teeth3ds-dataset-root -o /path/to/output/dir
```

The script will collect and test all the testing samples listed in the `*testing*` text files.

The default GPT model for tooth classification is `chatgpt-5.2`. If you would like to run with `doubao-seed 1.8` model, run with arguments:

```shell
python test_teeth3ds.py -d /path/to/teeth3ds-dataset-root -o /path/to/output/dir --gpt-model doubao-seed
```

Or, if you want no FDI classification, run:

```shell
python test_teeth3ds.py -d /path/to/teeth3ds-dataset-root -o /path/to/output/dir --disable-gpt
```


## Predict Single File

Use the code in `predictor.py`, or run with `python`:

```python
from predictor import Predictor

predictor = Predictor()
predictor.predict("/path/to/model/file")
predictor.visualize("/path/to/visualize/file")
predictor.export_json("/path/to/predicted/json")
```
