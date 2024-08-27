# Rise by Lifting Others: Interacting Features to Uplift Few-Shot Fine-Grained Classification
Few-shot fine-grained classification entails notorious subtle inter-class variation. Recent works address this challenge by developing attention mechanisms, such as the task discrepancy maximization (TDM) that can highlight discriminative channels. This paper, however, aims to reveal that, besides designing sophisticated attention modules, a well-designed input scheme, which simply blends two types of features and their interactions capturing different properties of the target object, can also greatly promote the quality of the learnt weights. To illustrate, we design a bi-feature interactive TDM (BiFI-TDM) module to serve as a strong foundation for TDM to discover the most discriminative channels with ease. Specifically, we design a novel mixing strategy to produce four sets of channel weights with different focuses, reflecting the properties of the corresponding input features and their interactions, as well as a proper feature re-weighting scheme. Extensive experiments on four benchmark fine-grained image datasets showcase superior performance of BiFI-TDM in metric-based few-shot methods.

# Approach
![BiFI-TDM](./imgs/mainFigure.png)
# Data Preparing
1. First, you need to set the value of `data_path` in the `config.yml` file to your dataset path.
2. The following datasets are used in our paper:
   - `CUB_fewshot_crop`: 100/50/50 classes for train/validation/test, using bounding-box cropped images as input.
   - `Stanford-Dogs`: 60/30/30 classes for train/validation/test.
   - `Stanford-Cars`: 130/17/49 classes for train/validation/test.
   - `Oxford-Flowers`: 51/26/25 classes for train/validation/test.

# Model Training and Testing
Our BiFI-TDM can be attached to any metric-based few-shot models. Our paper and code take ProtoNet and FRN as examples of metric modules.

To train a model with ProtoNet as the metric module from scratch using the `CUB` dataset, you can navigate to the `/Experiments/cub_crop/BiFI-TDM-Proto/ResNet-12_1-shot` subfolder. This folder contains two files: `train.py ` and `train.sh`. Running the shell script `train.sh` will train and evaluate the model using hyperparameters that match our paper. The explanations for these hyperparameters can be found in `trainers/trainer.py`.
```
cd /Experiments/cub_crop/BiFI-TDM-Proto/ResNet-12_1-shot
sh train.sh
```
To train a model with FRN as the metric module, you can run the following command:
```
cd /Experiments/cub_crop/BiFI-TDM-FRN/ResNet-12
sh train.sh
```
Model evaluation is based on the code `tm.evaluate(model)`(the last line) in `train.py`. You can also create a separate file dedicated to model testing.
# Few-shot Image Classification
## Fine-grained Few-shot Image Classification
![Fine-grained](./imgs/table2.png)
