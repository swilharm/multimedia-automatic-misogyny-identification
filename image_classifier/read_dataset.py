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
    test_labels = pd.read_csv('test_labels.txt', 
                              sep='\t', header=None, 
                              names=['file_name', 'misogynous', 'shaming', 'stereotype', 'objectification', 'violence'])
    test_labels = test_labels[['file_name', 'misogynous']]
    test_labels.to_csv('./data/test/labels.csv', index=None, sep='\t')
    
    train_labels = pd.read_csv('./data/TRAINING/training.csv', sep='\t')
    train_labels = train_labels[['file_name', 'misogynous']]
    train_labels.to_csv('./data/TRAINING/labels.csv', index=None, sep='\t')
    
    return train_labels, test_labels
    

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
    


def prepare_data(batch_size=16):

    train_labels, test_labels = extract_image_data()
    transforms = transform.Compose([
    transform.ToPILImage(),
    transform.Resize((256,256)),
    transform.ToTensor()])

    train_dataset = MamiDataset(train_labels, './data/TRAINING/', transforms)
    test_dataset = MamiDataset(test_labels, './data/test/', transforms)

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size,
                                  shuffle=True)

    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=batch_size,
                                 shuffle=True)
    
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    uzip_files()
    extract_image_data()
