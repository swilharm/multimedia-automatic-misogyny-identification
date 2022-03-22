from modules import *
from read_dataset import prepare_data, transform_data

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
_, test_dataloader = prepare_data(batch_size=16)

ckp_path = "./wide_resnet.pt"

net = models.wide_resnet50_2(pretrained=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 256)
net.fc = net.fc.cuda() if use_cuda else net.fc

if use_cuda:
    trained_model = torch.load(ckp_path)
else:
    trained_model = torch.load(ckp_path, map_location=torch.device('cpu'))
net.load_state_dict(trained_model)
net = net.cuda() if use_cuda else net
net.eval()


train_dataset, test_dataset = transform_data()

def visualize_model(net, num_images=4):
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))
    
    for i, data in enumerate(test_dataloader):
        print(i)
        inputs, labels = data
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net(inputs)
        _, preds = torch.max(outputs.data, 1)
        preds = preds.cpu().numpy() if use_cuda else preds.numpy()
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(2, num_images//2, images_so_far)
            ax.axis('off')
            ax.set_title('prediction: {}'.format(test_dataset.labels[preds[j]]))
            inputs = inputs.cpu()
            
            imshow(inputs[j].permute(1, 2, 0))
            
            if images_so_far == num_images:
                return plt.show()


if __name__ == "__main__":

    plt.ion()
    visualize_model(net)
    plt.ioff()
