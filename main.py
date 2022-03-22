from modules import *
from read_dataset import unzip_files
from resnet import train as resnet_train
from resnet import load_resnet
from bert import start_training as bert_train
from visualize import visualize_model
from combined import combined_evaluation


if __name__ == "__main__":
    unzip_files()
    print("Training image classifier...")
    resnet_train()
    print("Training text classifier...")
    bert_train()
    acc, f1 = combined_evaluation()
    print('Accuracy of the combination of models is: ', acc*100, '%') # 62.8%
    print('Macro f1-score of the combination of models is: ', f1) # 58.2
    
    