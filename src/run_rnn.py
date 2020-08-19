from main_lib import *

###############################################################################
#           Run Convolutional Neaural Network
###############################################################################
# Hyper Parameters
EPOCH = 40                # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 40
TIME_STEP = 300           # rnn time step / image height
INPUT_SIZE = 1            # rnn input size / image width
LR = 0.0001               # learning rate

rnn = RNN(INPUT_SIZE)
print(rnn)

train_dataset = EkgDataSet(1, '../1beat/1beat/')
test_dataset = EkgDataSet(0, '../1beat/1beat/')

print(train_dataset.__len__())
print(test_dataset.__len__())

MODEL_STORE_PATH = './Model/'

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()


###############################################################################
#           Run Recurrent Neaural Network
###############################################################################
loss_list = []
acc_list = []
total_step = len(train_loader)
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data
        b_x = b_x.view(-1, 300, 1)              # reshape x to (batch, time_step, input_size)

        output = rnn(b_x)                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        loss_list.append(loss.item())
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        # Track the accuracy
        total = b_y.size(0)
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == b_y).sum().item()
        acc_list.append(correct / total)

        if (step + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, EPOCH, step + 1, total_step, loss.item(),
                          (correct / total) * 100))

    test_rnn(rnn, test_loader)

# Save the model and plot
torch.save(rnn.state_dict(), MODEL_STORE_PATH + 'conv_net_model_rnn_2.ckpt')

p = figure(y_axis_label='Loss', width=850, y_range=(0, 1), title='PyTorch ConvNet results')
p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
p.line(np.arange(len(loss_list)), loss_list)
p.line(np.arange(len(loss_list)), np.array(acc_list) * 100, y_range_name='Accuracy', color='red')
show(p)
