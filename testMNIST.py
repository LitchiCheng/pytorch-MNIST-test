import torch
import torch.nn as nn
import torchvision.datasets
import matplotlib.pyplot as plt
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 连接序列
        self._conv1_layer = nn.Sequential(
            # 卷积
            nn.Conv2d(1,15,5),
            # 激活函数
            nn.ReLU(),
            # 最大池化，减少特征量，选特征最大的数，是一种下采样
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self._conv2_layer = nn.Sequential(
            nn.Conv2d(15,30,5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self._full_layer = nn.Sequential(
            # 卷积层都是四维张量，展平为二维张量给连接层用
            nn.Flatten(),
            nn.Linear(in_features=480, out_features=60),
            nn.ReLU(),
            nn.Linear(in_features=60, out_features=10),
        )
    
    def forward(self, input):
        # 层层连接，两个卷积层，最后全连接层
        output = self._conv1_layer(input)
        output = self._conv2_layer(output)
        output = self._full_layer(output)
        return output

class Test:
    def __init__(self):
        # MNIST数据集，用于训练，一次抓60 size
        self._train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data/', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                    ])),
            batch_size=60, shuffle=True)
        # 用于测试，一次抓500 size
        self._test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data/', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                    ])),
            batch_size=500, shuffle=True)
        # 训练次数
        self._epochs = 3
        self._cnn = CNN()
        # 交叉熵损失函数，刻画的是两个概率分布的距离，交叉熵越小，概率分布越接近
        self._loss_func = nn.CrossEntropyLoss()
        # 优化器
        self._optim = torch.optim.Adam(self._cnn.parameters(), lr=0.01)
        if torch.cuda.is_available():
            print("Use CUDA training!")
            self._device = torch.device("cuda")
        else:
            print("Use CPU training!")
            self._device = torch.device("cpu")
        
    def train(self):
        loss_d = []
        for epoch in range(1, self._epochs + 1):
            self._cnn.train(mode=True)
            for idx, (train_img, train_label) in enumerate(self._train_loader):
                # 复制到device中
                train_img = train_img.to(self._device)
                train_label = train_label.to(self._device)
                outputs = self._cnn(train_img)
                # 清除梯度
                self._optim.zero_grad()
                loss = self._loss_func(outputs, train_label)
                # 反向传播  
                loss.backward()
                # 更新权重
                self._optim.step()
                # print('Train epoch {}: loss: {:.6f}'.format(epoch,loss.item()))
                loss_d.append(loss.item())
        plt.plot(range(0,len(loss_d)),loss_d)
        plt.show()

    def test(self):
        correct_num = 0
        total_num = 0
        loss_d = []
        self._cnn.train(mode=False)

        with torch.no_grad():
            for idx, (test_img, test_label) in enumerate(self._test_loader):
                test_img = test_img.to(self._device)
                test_label = test_label.to(self._device)

                total_num += test_label.size(0)

                outputs = self._cnn(test_img)
                loss = self._loss_func(outputs, test_label)
                loss_d.append(loss.item())

                predictions = torch.argmax(outputs, dim=1)
                correct_num += torch.sum(predictions == test_label)
        acc_num = ((correct_num.item()/total_num)*100)
        title_str ="Accuracy:"+str(acc_num)+"%"
        plt.title(title_str)
        plt.plot(range(0,len(loss_d)),loss_d)
        plt.show()
            
    def plotTestResult(self):
        iteration = enumerate(self._test_loader)
        idx, (test_img, test_label) = next(iteration)

        with torch.no_grad():
            outputs = self._cnn(test_img)

            fig = plt.figure()
            for i in range(4 * 2):
                plt.subplot(4, 2, i + 1)
                plt.tight_layout()
                plt.imshow(test_img[i][0], cmap='gray', interpolation='none')
                plt.title('real: {}, predict: {}'.format(
                    test_label[i], outputs.data.max(1, keepdim=True)[1][i].item()
                ))
                plt.xticks([])
                plt.yticks([])
            plt.show()

    def savePthModel(self, pth_name:str):
        torch.save(self._cnn.state_dict(), pth_name)

    def saveOnnxModel(self, onnx_name:str):
        input = torch.randn(1,1,28,28)
        torch.onnx.export(self._cnn, input, onnx_name, verbose=True)

    
if __name__ == "__main__":
    mt = Test()
    mt.train()
    mt.test()
    mt.plotTestResult()
    mt.savePthModel("model.pth")
    mt.saveOnnxModel("model.onnx")