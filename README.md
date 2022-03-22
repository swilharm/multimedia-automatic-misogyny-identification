# Task 5: Multimedia Automatic Misogyny Identification (MAMI)
#### Team MAMI-SAN: Sebastian Wilharm, Aleksandra Sharapova, Nailia Mirzakhmedova

1. Create a new conda environment with all required dependencies: `conda env create -f environment.yml`
2. Activate the environment: `conda activate mami-san`
3. Download dataset [here](https://drive.google.com/drive/folders/1x04eqdhH_JBadUeutIf02szK_778mmHH) and put all files in folder ./MAMI DATASET/
4. Download resnet weights [here](https://drive.google.com/file/d/1Ln6hVyvePq1OeWYgVyTOdaY63FU5HOFh/view?usp=sharing) and put them in the main directory
5. Download bert weights [here](https://drive.google.com/file/d/1qILWpYfbNouY6ScOQEb1K3hLSGFAA3yi/view?usp=sharing) and put them in the main directory
6. To run everything with one command, run `python main.py`. Alternatively go through it step by step as described below.

### Model Desription

The model that was used for image classification is the pretrained Wide ResNet-50-2 model from [“Wide Residual Networks”](https://arxiv.org/pdf/1605.07146.pdf).

ResNet weights can be found [here](https://drive.google.com/file/d/1Ln6hVyvePq1OeWYgVyTOdaY63FU5HOFh/view?usp=sharing)

It achieved the accuracy of 65.8% on the test set and 75.9% on the training set just after 3 epochs. 

The model that was used for text classification is the pretrained cased Bidirectional Encoder Representations from Transformers  ([BERT](https://arxiv.org/abs/1810.04805)), which achieves 53.8% accuracy on the test set after 3 epochs.

BERT weights can be found [here](https://drive.google.com/file/d/1qILWpYfbNouY6ScOQEb1K3hLSGFAA3yi/view?usp=sharing)

The combined accuracy of the two models on the whole test dataset is 62.8% and the macro averaged f1 score is 58.2.

### Data Preprocessing

To preprocess the dataset, run the following command once:

```
python read_dataset.py 
```

The command unzips the data into the ```data``` folder in the working folder (creates it if necessary), preprocesses the files and creates dataloaders for the training and testing.
Two files in the test and train folders are created under the name ```labels.csv```.

### Training

To run the training script for image classifier, execute:

```
python resnet.py
```


To run the training script for text classifier, execute:

```
python bert.py
```


### Evaluation 

To load the image classifier weights and do the evaluaton for 4 random images from the test set, run the following command:
```
python vizualize.py
```

To load both classifiers and compute their accuracy, run:

```
python combined.py
```

### Notebook

Data preprocessing, exploration and visualization, as well as all the code from the scripts with outputs can be found in the Jupyter Notebook ```bert_resnet.ipynb```