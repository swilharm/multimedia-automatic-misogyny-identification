### Model Desription

The model that was used for the training is pretrained Wide ResNet-50-2 model from [“Wide Residual Networks”](https://arxiv.org/pdf/1605.07146.pdf).

Model weights can be found [here](https://drive.google.com/file/d/1Ln6hVyvePq1OeWYgVyTOdaY63FU5HOFh/view?usp=sharing)

It achieved the accuracy of 65.8% on the test set and 75.9% on the training set just after 3 epochs. 


### Data Preprocessing

To preprocess the dataset, run the following command once:

```
python read_dataset.py 
```

The command unzips the data into the ```data``` folder in the working folder (creates it if necessary), preprocesses the files and creates dataloaders for the training and testing.
Two files in the test and train folders are created under the name ```labels.csv```.

### Training

To run the training script, execute:

```
python main.py
```

### Evaluation 

To load the model weights and do the evaluaton for 4 random images from the test set, run the following command:
```
python vizualize.py
```
