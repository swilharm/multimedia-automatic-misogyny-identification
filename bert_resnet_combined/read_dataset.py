from modules import * 

def unzip_files():
    path = './data/'
    try:
        os.mkdir(path)
    except OSError as error:
        pass
          
    with zipfile.ZipFile('MAMI DATASET-20220117T151001Z-001.zip', 'r') as zip_ref:
            zip_ref.extractall('./', pwd=b'*MaMiSemEval2022!')
    
    zip_list = ['./MAMI DATASET/training.zip', './MAMI DATASET/test.zip']
    for zip in zip_list:
        with zipfile.ZipFile(zip, 'r') as zip_ref:
            zip_ref.extractall('./data/', pwd=b'*MaMiSemEval2022!')

def extract_image_data(): 

            
    data = pd.read_csv('./data/TRAINING/training.csv', sep='\t')
    test_labels = pd.read_csv('./MAMI DATASET/test_labels.txt', 
                              sep='\t', header=None, 
                              names=['file_name', 'misogynous', 'shaming', 'stereotype', 'objectification', 'violence'])
    test_labels = test_labels[['file_name', 'misogynous']]
    test_labels.to_csv('./data/test/labels.csv', index=None, sep='\t')
    
    train_labels = pd.read_csv('./data/TRAINING/training.csv', sep='\t')
    train_labels = train_labels[['file_name', 'misogynous']]
    train_labels.to_csv('./data/TRAINING/labels.csv', index=None, sep='\t')
    
    return train_labels, test_labels
    
def extract_text_data():
    train_df = pd.read_csv('./data/TRAINING/training.csv', sep='\t')
    train_df.rename(columns={'Text Transcription': 'TextTranscription'}, inplace=True)
    train_df = train_df.drop(train_df.columns[[2,3,4,5]], axis=1)
    
    test_labels = pd.read_csv('./MAMI DATASET/test_labels.txt', 
                              sep='\t', header=None, 
                              names=['file_name', 'misogynous', 'shaming', 'stereotype', 'objectification', 'violence'])
    test_labels = test_labels[['file_name', 'misogynous']]
    test_df = pd.read_csv('./data/test/Test.csv', sep='\t')
    test_df.rename(columns={'Text Transcription': 'TextTranscription'}, inplace=True)
    test_df['misogynous'] = test_labels['misogynous']
    
    return train_df, test_df
    

class MamiDataset(Dataset):
    
    def __init__(self, csv, img_folder, transform):
        self.csv = csv
        self.transform = transform
        self.img_folder = img_folder
    
        self.image_names = self.csv[:]['file_name']
        self.labels = np.array(self.csv.drop(['file_name'], axis=1))
    
    # The __len__ function returns the number of samples in our dataset
    def __len__(self):
    
        return len(self.image_names)

    
    def __getitem__(self,idx):
    
        image = cv2.imread((self.img_folder + self.image_names).iloc[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        image = self.transform(image)
        targets = self.labels[idx]
  

        return (image, targets)
    


def transform_data():
    
    train_labels, test_labels = extract_image_data() 

    train_transforms = transform.Compose([
        transform.ToPILImage(),
        transform.RandomRotation(25),
        transform.Resize((256,256)),
        transform.CenterCrop(224),
        # transform.RandomHorizontalFlip(p=0.7),
        transform.ColorJitter(),
        transform.ToTensor()])

    test_transforms = transform.Compose([transform.ToPILImage(),
                                     transform.Resize((256,256)),
                                     transform.CenterCrop(224),
                                     transform.ToTensor()])

    train_dataset = MamiDataset(train_labels, './data/TRAINING/', train_transforms)
    test_dataset = MamiDataset(test_labels, './data/test/', test_transforms)
    return train_dataset, test_dataset

def prepare_data(batch_size=16):
    train_dataset, test_dataset = transform_data()
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size,
                                  shuffle=True)

    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=batch_size,
                                 shuffle=False)
    
    return train_dataloader, test_dataloader


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

def prepare_text_data(batch_size=16):
    train_df, test_df = extract_text_data()
    train, test = Dataset(train_df), Dataset(test_df)
    train_dataloader_text = DataLoader(train, batch_size=16, shuffle=True)
    test_dataloader_text = DataLoader(test, batch_size=16, shuffle=False)
    
    return train_dataloader_text, test_dataloader_text

if __name__ == "__main__":
    uzip_files()
    # extract_image_data()
    # extract_text_data()
