import torch
import torch.nn as nn 
import matplotlib.pyplot as plt 

from utils import ALL_LETTERS, N_LETTERS
from utils import load_data, line_to_tensor, training_example_i

learning_rates = [1, 0.5, 0.1, 0.05, 0.01]
n_examleperclass = 300
n_iter = 100
criterion = nn.NLLLoss() 

category_lines, all_categories = load_data() # list(category_lines.keys()) == all_categories

n_categories = len(all_categories)



def category_from_output(output):
    return torch.argmax(output, dim=1).item()


class MyRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()


        self.hsize = hidden_size
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, input, hidden):
        combine = torch.cat((input, hidden), dim=1)
        newhidden = torch.sigmoid(self.i2h(combine))
        combine = torch.cat((input, newhidden), dim=1)
        output = self.softmax(torch.sigmoid(self.i2o(combine)))

        return output, newhidden
    
    def hidinit(self):
        return torch.zeros(1, self.hsize)



def oneepoch(categories_lines, all_categories):
    sum_losses = 0 # to do the mean of the losses
    counter = 0 #counter for number of loss added for the mean
    for cat in all_categories: #n, cat in enumerate(all_categories): # for every categories
        for i in range(len(category_lines[cat])): # for every instance
            category_tensor, line_tensor = training_example_i(categories_lines, all_categories, cat, i)

            hidden = model.hidinit()
            for j in range(line_tensor.shape[0]): #run the rnn
                output, hidden = model(line_tensor[j], hidden)

 
            loss = criterion(output, category_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # optimize 

            sum_losses += loss.item()
            counter +=1

            if i <=n_examleperclass: # speed up calculation for practice
                break

        #print(f"Done {n+1}/{len(all_categories)}")

    return sum_losses/counter

            


for lr in learning_rates:
    loss_graph = list()

    model = MyRNN(N_LETTERS, 100, n_categories)
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    for epoch in range(n_iter):
        loss = oneepoch(category_lines, all_categories)

        if (epoch) % 5 == 0:
            print(f"{epoch+1} done.")
            loss_graph.append(loss)


    plt.plot(loss_graph, label = f"{lr}")

plt.legend()
plt.show()


    