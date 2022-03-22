from modules import *
from read_dataset import extract_text_data, prepare_text_data

train_df, test_df = extract_text_data()
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
_, test_dataloader_text = prepare_text_data(batch_size=16)

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        transformers.logging.set_verbosity_error()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

transformers.logging.set_verbosity_error()
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
labels = {'0':0,
          '1':1
          }

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [labels[str(label)] for label in df['misogynous']]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['TextTranscription']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=16, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=16)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr= learning_rate)

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            train_loss = 0.0
            train_acc = 0.0
            val_loss = 0.0
            val_acc = 0.0

            model.train()

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                
                batch_loss = criterion(output, train_label)
                train_loss += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                train_acc += acc
                
                model.zero_grad()
                batch_loss.backward()
                optimizer.step()

            model.eval()

            for val_input, val_label in val_dataloader:

                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                val_loss += batch_loss.item()
                    
                acc = (output.argmax(dim=1) == val_label).sum().item()
                val_acc += acc
            
            train_loss = train_loss/len(train_dataloader.dataset)
            val_loss = val_loss/len(val_dataloader.dataset)

        # print training/validation statistics 
            print('Epoch: {} \tTraining Loss: {:.6f} \tTrain Acc: {:.6f} \tValidation Loss: {:.6f} \tValidation Acc: {:.6f}'.format(
            epoch_num, 
            train_loss,
            train_acc,
            val_loss,
            val_acc
            ))
            
            checkpoint = {
            'epoch': epoch_num + 1,
            'valid_losstarget = target.reshape(-1)_min': val_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),}
        
        # save checkpoint
            torch.save(checkpoint, './checkpoint/')            
    # return trained model
    return model
   
def evaluate(model, test_data, finetuned=True):
    if finetuned:
      model = load_bert()
    print('Running evaluation on the test set...')

    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=16)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)

              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc
    
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

def pytorch_predict_text(model, test_loader, device):
    '''
    Make prediction from a pytorch model 
    '''
    # set model to evaluate model
    model.eval()
    
    y_true = torch.tensor([], dtype=torch.long, device=device)
    all_outputs = torch.tensor([], device=device)
    
    # deactivate autograd engine and reduce memory usage and speed up computations
    with torch.no_grad():
        for test_input, test_label in test_loader:
            
            label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input = test_input['input_ids'].squeeze(1).to(device)
            
            output = model(input, mask)
            y_true = torch.cat((y_true, label), 0)
            all_outputs = torch.cat((all_outputs, output), 0)
    
    y_true = y_true.cpu().numpy()  
    _, y_pred = torch.max(all_outputs, 1)
    y_pred = y_pred.cpu().numpy()
    y_pred_prob = F.softmax(all_outputs, dim=1).cpu().numpy()
    
    return y_true, y_pred, y_pred_prob
    

def load_bert(checkpoint='./bert.pt'):
    bert = BertClassifier()
    transformers.logging.set_verbosity_error()
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(bert.parameters(), lr=1e-6)
    use_cuda = torch.cuda.is_available()
    bert = bert.cuda() if use_cuda else bert

    if use_cuda:
        trained_bert = torch.load(checkpoint)
    else:
        trained_bert = torch.load(checkpoint, map_location=torch.device('cpu'))
    bert.load_state_dict(trained_bert)
    true_txt, pred_txt, prob_txt = pytorch_predict_text(bert, test_dataloader_text, device)
    return true_txt, pred_txt, prob_txt

def start_training():
    np.random.seed(1)
    df_train, df_val = np.split(train_df.sample(frac=1, random_state=42), [int(.99*len(train_df))])
    bert = BertClassifier()
    transformers.logging.set_verbosity_error()
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(bert.parameters(), lr=1e-6)
    use_cuda = torch.cuda.is_available()
    bert = bert.cuda() if use_cuda else bert
    EPOCHS = 1
    model = BertClassifier()
    LR = 1e-6            
    train(model, df_train, df_val, LR, EPOCHS)


if __name__ == "__main__":
  start_training()
