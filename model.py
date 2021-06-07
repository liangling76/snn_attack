import torch
import torch.nn as nn
import torch.nn.functional as F

thresh = 0.3
lens = 0.5
probs = 0.25
decay = 0.25
batch_size = 40


class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()


act_fun = ActFun.apply


def mem_update(conv, x, mem, spike):
    mem = mem * decay * (1. - spike) + conv(x)
    spike = act_fun(mem)
    return mem, spike


def mem_update_pool(opts, x, mem, spike):
    mem = mem * decay * (1. - spike) + opts(x,2)
    spike = act_fun(mem)
    return mem, spike


class MNIST(nn.Module):

    def __init__(self):
        super(MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1,    128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128,  256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256,  512, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(7 * 7 * 512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512,  10)


    def forward(self, input, time_window=15):
        c1_mem = c1_spike = torch.zeros(batch_size, 128, 28, 28).cuda()
        c2_mem = c2_spike = torch.zeros(batch_size, 256, 28, 28).cuda()
        p2_mem = p2_spike = torch.zeros(batch_size, 256, 14, 14).cuda()

        c3_mem = c3_spike = torch.zeros(batch_size, 512, 14, 14).cuda()
        p3_mem = p3_spike = torch.zeros(batch_size, 512, 7, 7).cuda()

        c4_mem = c4_spike = torch.zeros(batch_size, 1024, 7, 7).cuda()
        c5_mem = c5_spike = torch.zeros(batch_size,  512, 7, 7).cuda()

        h1_mem = h1_spike = torch.zeros(batch_size, 1024).cuda()
        h2_mem = h2_spike = torch.zeros(batch_size, 512).cuda()
        h3_mem = h3_spike = h3_sumspike = torch.zeros(batch_size, 10).cuda()

        for step in range(time_window):
            x = input > torch.rand(input.size()).cuda()
            x = x.float()

            c1_mem, c1_spike = mem_update(self.conv1,        x,        c1_mem, c1_spike)
            x = F.dropout(c1_spike, p=probs, training=self.training)

            c2_mem, c2_spike = mem_update(self.conv2,        x,        c2_mem, c2_spike)
            p2_mem, p2_spike = mem_update_pool(F.avg_pool2d, c2_spike, p2_mem, p2_spike)
            x = F.dropout(p2_spike, p=probs, training=self.training)

            c3_mem, c3_spike = mem_update(self.conv3,        x,        c3_mem, c3_spike)
            p3_mem, p3_spike = mem_update_pool(F.avg_pool2d, c3_spike, p3_mem, p3_spike)
            x = F.dropout(p3_spike, p=probs, training=self.training)

            c4_mem, c4_spike = mem_update(self.conv4, x, c4_mem, c4_spike)
            x = F.dropout(c4_spike, p=0.5, training=self.training)

            c5_mem, c5_spike = mem_update(self.conv5, x, c5_mem, c5_spike)
            x = F.dropout(c5_spike, p=probs, training=self.training)

            x = x.view(batch_size, -1)

            h1_mem, h1_spike = mem_update(self.fc1, x,        h1_mem, h1_spike)
            x = F.dropout(h1_spike, p=probs, training=self.training)

            h2_mem, h2_spike = mem_update(self.fc2, x,        h2_mem, h2_spike)
            h3_mem, h3_spike = mem_update(self.fc3, h2_spike, h3_mem, h3_spike)
            h3_sumspike += h3_spike

        outputs = h3_sumspike / time_window
        return outputs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.05)
                nn.init.constant_(m.bias, 0)

