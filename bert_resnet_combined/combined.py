from modules import *
from main import load_resnet
from bert import load_bert


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load image model
true_im, pred_im, prob_im  = load_resnet(checkpoint='./wide_resnet.pt')
true_txt, pred_txt, prob_txt = load_bert(checkpoint='./bert.pt')

labels = []
for i in range(len(true_txt)):
    if pred_txt[i] or pred_im[i] == 1:
        labels.append(1)
else:
  labels.append(0)
  labels = np.array(labels)
acc = sum(labels == true_txt)/len(true_txt)

if __name__ == "__main__":
    print('Accuracy of the combination of models is: ', acc*100, '%')
    # 62.8%
