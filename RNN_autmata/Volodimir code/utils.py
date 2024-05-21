# path = "/home/volodimir/Bureau/ForLang/names.txt"
import torch


def lang_names(path):
    names = []
    extensions = []
    with open(path) as file:
        length = len(file.readlines())
    with open(path) as file:
        A = True
        h = 0
        while A:
            line = file.readline()
            h+=1
            A = not(line=='\n')
            if A:
                names.append(line[:-1])
        for j in range(length-h):
            line = file.readline()
            extensions.append(line[:-1])
    return (names,extensions)

# names, ext = lang_names(path)

class Loss_for_saturation(torch.autograd.Function):

    def forward(ctx, y_pred, y):
        criterion = torch.nn.CrossEntropyLoss()
        ctx.save_for_backward(y_pred, y)
        res = criterion(y_pred,y)
        return res
    
    def backward(ctx, grad_output):
        def reggy(tens):
            return (torch.sign(tens))*(((tens+1)**(1.38))/torch.log(tens+1) - 1.3)
        
        y_pred, y = ctx.saved_tensors
        # grad_input = torch.mean( -2.0 * (y - y_pred)).repeat(y_pred.shape[0]) 
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(y_pred,y)
        loss.requires_grad = True
        grad = loss.backward()
        print(grad)
        print("yaye")
        grad[0<=grad<0.001] = 0.01
        grad[0>=grad>-0.001] = -0.01
        res = reggy(grad)

        return res, None
