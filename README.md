# ClmPred: Contrastive Learning based Myopia Prediction from Retinal Fundus Images
This is our implementation of ClmPred.

## Abstract

> Myopia as a prevalent issue of global public health concern, urgently requires the integration of various modal data to establish effective early detection methods. In this paper, we construct a Retinal Fundus Myopia Prediction dataset, named RFMPred. Meanwhile, to address the scarcity of fundus image data in the early stages of myopia, we integrate existing public fundus image datasets and build a contrastive learning-based myopia prediction model, named ClmPred, and reveal key regions of the fundus in the early stages of myopia through spherical equivalent refraction prediction tasks. Experimental results demonstrate that the model achieves root mean square error, Pearson correlation coefficient, and mean absolute error of 1.630, 0.672, and 1.321 on the RFMPred dataset, respectively. Furthermore, through a comprehensive analysis of the experimental results, we find that the regions associated with visual acuity in the early stages of myopia include the optic disc, optic cup, macula central fovea, and vascular areas, which can be utilized in optometry instruments.

## Environment
* Python >=3.5
* PyTorch >=1.5.1
* CUDA >= 10.2
* sklearn >=1.0.2
* matplotlib
* numpy
* shutil
* pandas

## Installation

You can install via conda environment .yaml file

```bash
conda env create -f env.yml
conda activate ClmPred
```


## Feature Extraction Backbone Train(the trained model will be saved at ./upstreamtrain/checkpoint_0600.pth)
Use the script `run.py` to train a model in the our dataset:
``` bash
cd ClmPred
python run.py --savepath upstreamtrain


```
## Finetune and Test(The 5-fold cross-validation results will be saved at ./results)
``` bash
cd ClmPred   
python ./finetune/runforKfold_early.py --savepath results
```


## License

This project is open sourced under MIT license.
