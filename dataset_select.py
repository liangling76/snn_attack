import torch
import torchvision
import torchvision.transforms as transforms

batch = 64
test_set = torchvision.datasets.MNIST(root= './data/mnist', train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch, shuffle=False, num_workers=0)

num_select = 5

test_image = torch.Tensor(10 * num_select, 1, 28, 28)
test_label = torch.Tensor(10 * num_select)
count_test = [0 for _ in range(10)]

# load testing dataset
for image, label in test_loader:
    for i in range(batch):
        label_item = label[i].item()
        if count_test[label_item] < num_select:
            idx = label_item * num_select + count_test[label_item]
            test_image[idx] = image[i]
            test_label[idx] = label_item

            count_test[label_item] += 1

    if sum(count_test) == num_select * 10:
        break

num_total = num_select * 10
torch.save(test_image, './ckpt/test_image_' + str(num_total) + '.pth.tar')
torch.save(test_label, './ckpt/test_label_' + str(num_total) + '.pth.tar')


