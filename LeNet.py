import torch
from d2l import torch as d2l
from torch import nn
import  os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Reshape(torch.nn.Module):
    def forward(self,x):
        return x.view(-1,1,28,28)

net=torch.nn.Sequential(Reshape(),
                  nn.Conv2d(1,6,kernel_size=5,padding=2),nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2,stride=2),
                  nn.Conv2d(6,16,kernel_size=5),nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2,stride=2),
                  nn.Flatten(),
                  nn.Linear(16*5*5,120),nn.ReLU(),
                  nn.Linear(120,84),nn.ReLU(),
                  nn.Linear(84,10),nn.Softmax(dim=1)
                  )
x=torch.randn(size=(1,1,28,28),dtype=torch.float32)

for layer in net:
    x=layer(x)
    print(layer.__class__.__name__,'output shape:\t',x.shape)

batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size=batch_size)

def evaluate_accuracy_gpu(net,data_iter,device=None):
    net.eval()
    if not device:
        device=next(iter(net.parameters())).device
    metric=d2l.Accumulator(2)
    for X,y in data_iter:
        X,y=X.to(device),y.to(device)
        metric.add(d2l.accuracy(net(X),y),d2l.size(y))
    return metric[0]/metric[1]

def train_ch76(net,train_iter,test_iter,lr,num_epochs,device=d2l.try_gpu()):
    def init_ini(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
    net.apply(init_ini)
    net.to(device)
    optimizer=torch.optim.SGD(net.parameters(),lr=lr)
    loss=nn.CrossEntropyLoss()
    timer=d2l.Timer()
    for epoch in range(num_epochs):
        metric=d2l.Accumulator(3)
        for i,(X,y) in enumerate(train_iter):
            timer.start()
            net.train()
            optimizer.zero_grad()
            X,y=X.to(device),y.to(device)
            y_hat=net(X)
            l=loss(y_hat,y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0],d2l.accuracy(y_hat,y),X.shape[0])
            timer.stop()
            test_acc=evaluate_accuracy_gpu(net,test_iter)
            train_loss=metric[0]/metric[2]
            train_acc=metric[1]/metric[2]
    print(f'loss{train_loss:.3f},trainacc{train_acc:.3f},testacc{test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(device)}')

lr, num_epochs = 0.003, 10
train_ch76(net, train_iter, test_iter, lr,num_epochs)
