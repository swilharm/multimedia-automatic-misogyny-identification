from modules import *
from resnet import load_resnet
from bert import load_bert

def combined_evaluation():

    # load checkpoints
    print("Loading resnet checkpoint...")
    true_im, pred_im, prob_im  = load_resnet(checkpoint='./wide_resnet.pt')
    print("Loading bert checkpoint...")
    true_txt, pred_txt, prob_txt = load_bert(checkpoint='./bert.pt')

    print("Evaluating model combination...")
    labels = []
    for i in range(len(true_txt)):
        if pred_txt[i] or pred_im[i] == 1:
            labels.append(1)
    else:
      labels.append(0)
      labels = np.array(labels)
    acc = sum(labels == true_txt)/len(true_txt)
    print(acc)
    print(true_txt)
    print(labels)
    f1 = f1_score(true_txt, labels, average='macro')
    return acc, f1

if __name__ == "__main__":
    acc, f1 = combined_evaluation()
    print('Accuracy of the combination of models is: ', acc*100, '%') # 62.8%
    print('Macro f1-score of the combination of models is: ', f1) # 58.2
