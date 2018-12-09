# Author: Guo-qing Jiang (jianggq@mit.edu)
# Pytorch second oder gradient calculation for the diagonal of Hessian matrix 
# feel free to copy
import torch


def get_second_order_grad(grads, xs):
    start = time.time()
    grads2 = []
    for j, (grad, x) in enumerate(zip(grads, xs)):
        print('2nd order on ', j, 'th layer')
        print(x.size())
        grad = torch.reshape(grad, [-1])
        grads2_tmp = []
        for count, g in enumerate(grad):
            g2 = torch.autograd.grad(g, x, retain_graph=True)[0]
            g2 = torch.reshape(g2, [-1])
            grads2_tmp.append(g2[count].data.cpu().numpy())
        grads2.append(torch.from_numpy(np.reshape(grads2_tmp, x.size())).to(DEVICE_IDS[0]))
        print('Time used is ', time.time() - start)
    for grad in grads2:  # check size
        print(grad.size())

    return grads2

# datainput/model/optimizer setup is ommited here
optimizer.zero_grad()
xs = optimizer.param_groups[0]['params']
ys = loss  # put your own loss into ys

grads = torch.autograd.grad(ys, xs, create_graph=True)  # first order gradient

grads2 = get_second_order_grad(grads, xs)  # second order gradient
