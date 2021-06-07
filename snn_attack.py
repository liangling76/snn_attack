from model_attack import *
from torch.autograd.gradcheck import zero_gradients
import torch
import sys

attack_mode = 'untarget'  # untarget/target
thresh_origin_value = 0.3 # original firing threshold
thresh_attack_value = 1.0 # firing threshold of pernultimate layer
epsilon = 1               # control perturbation (0,1]
cw = 0.1                  # set to 0 if do not adopt cwL2


model = MNIST().cuda()
model.load_state_dict(torch.load('./ckpt/mnist_ce.pth.tar'))
model.eval()

criterion = torch.nn.CrossEntropyLoss().cuda()

class Attack():
    def __init__(self):
        self.succ = 0
        self.fail = 0
        self.wrong = 0
        self.avg_diff = 0

        self.dataset_len = 0
        self.time_window = 15

        self.max_diff = epsilon  
        self.gt = 0.95
        self.num_iter = 25

        self.c = cw

    def sample_input(self, input):
        spike_input = torch.zeros(1, 1, 28, 28, self.time_window).cuda()
        for i in range(self.time_window):
            spike_input[:, :, :, :, i] = input > torch.rand(input.size()).cuda()

        return spike_input

    def attack_untarget(self):
        dataset_image = torch.load('./ckpt/test_image_50.pth.tar')
        dataset_label = torch.load('./ckpt/test_label_50.pth.tar')
        self.dataset_len = dataset_image.shape[0]
        image_ones = torch.ones(1, 1, 28, 28, self.time_window).cuda()

        for i in range(self.dataset_len):
            input = dataset_image[i].cuda().unsqueeze_(0)
            org_input = input.clone()
            spike_input = self.sample_input(input)

            label = torch.autograd.Variable(torch.LongTensor([dataset_label[i]]).cuda(), requires_grad=False)
            label_value = dataset_label[i].item()

            set_thresh_pernultimate(thresh_origin_value)
            output = model(spike_input)
            _, pred = torch.max(output.cpu().data, 1)

            succ_flag = False
            if pred != label_value:
                self.wrong += 1
                continue

            for iter in range(self.num_iter):
                spike_input = torch.autograd.Variable(spike_input, requires_grad=True)
                zero_gradients(spike_input)

                set_thresh_pernultimate(thresh_attack_value)
                output = model(spike_input)
                loss = criterion(output, label)

                # perform attack
                loss.backward(retain_graph=True)
                grad = spike_input.grad
                grad_max = grad.abs().max()

                pre_spike_input = spike_input.clone()

                if grad_max != 0:
                    grad_abs = grad.abs()
                    grad_abs /= grad_abs.max()
                    grad_sample = torch.rand(grad.shape).cuda()
                    grad_abs[grad_abs >= grad_sample] = 1
                    grad_abs[grad_abs < grad_sample] = 0

                    grad_sign = torch.sign(grad)
                    grad_sign *= grad_abs

                    spike_input = spike_input.clone()
                    spike_input[grad_sign == 1] = 1
                    spike_input[grad_sign == -1] = 0

                else:
                    spike_input = spike_input.clone()
                    grad_mask = torch.rand(grad.shape).cuda() > self.gt
                    spike_input[grad_mask == 1] = image_ones[grad_mask == 1] - spike_input[grad_mask == 1]

                modi_pixel = torch.sum((spike_input - pre_spike_input), -1)
                modi_pixel /= self.time_window
                input_pre = input.clone().detach()
                input += modi_pixel
                input -= self.c * (input_pre - org_input)

                input[input > 1] = 1
                input[input < 0] = 0

                # sample new input
                spike_input = self.sample_input(input)
                set_thresh_pernultimate(thresh_origin_value)
                output = model(spike_input)
                _, pred = torch.max(output.cpu().data, 1)

                tmp_diff = torch.mean((org_input - input)**2).item()

                if pred != label_value and torch.sum(output == torch.max(output)) == 1 and tmp_diff <= self.max_diff:
                    self.succ += 1
                    succ_flag = True
                    self.avg_diff += tmp_diff

                if iter == self.num_iter - 1 or succ_flag or tmp_diff > self.max_diff:
                    if not succ_flag:
                        self.fail += 1
                    if (i + 1) % 25 == 0 and self.succ > 0:
                        print(str(i + 1) + '\tsucc: ' + str(self.succ) + '\tfail: ' + str(self.fail) +
                              '\twrong: ' + str(self.wrong) + '\tavg diff: ' + str(self.avg_diff / (1.0 * self.succ)))
                    break

    def attack_target(self):
        dataset_image = torch.load('./ckpt/test_image_50.pth.tar')
        dataset_label = torch.load('./ckpt/test_label_50.pth.tar')
        self.dataset_len = dataset_image.shape[0]
        image_ones = torch.ones(1, 1, 28, 28, self.time_window).cuda()

        for i in range(self.dataset_len):
            org_input = dataset_image[i].cuda().unsqueeze_(0)
            org_spike_input = self.sample_input(org_input)

            org_label = dataset_label[i]
            target_list = [j for j in range(10)]
            target_list.remove(org_label.item())

            set_thresh_pernultimate(thresh_origin_value)
            output = model(org_spike_input)
            _, pred = torch.max(output.cpu().data, 1)

            if pred != org_label.item():
                self.wrong += 1
                continue

            for j in target_list:
                input = org_input.clone()
                spike_input = org_spike_input.clone()
                label = torch.autograd.Variable(torch.LongTensor([j]).cuda(), requires_grad=False)

                succ_flag = False

                for iter in range(self.num_iter):
                    spike_input = torch.autograd.Variable(spike_input, requires_grad=True)
                    zero_gradients(spike_input)

                    set_thresh_pernultimate(thresh_attack_value)
                    output = model(spike_input)
                    loss = criterion(output, label)

                    # perform attack
                    loss.backward(retain_graph=True)
                    grad = spike_input.grad
                    grad_max = grad.abs().max()

                    pre_spike_input = spike_input.clone()

                    if grad_max != 0:
                        grad_abs = grad.abs()
                        grad_abs /= grad_abs.max()
                        grad_sample = torch.rand(grad.shape).cuda()
                        grad_abs[grad_abs >= grad_sample] = 1
                        grad_abs[grad_abs < grad_sample] = 0

                        grad_sign = torch.sign(grad)
                        grad_sign *= grad_abs

                        spike_input = spike_input.clone()
                        spike_input[grad_sign == 1] = 0
                        spike_input[grad_sign == -1] = 1

                    else:
                        spike_input = spike_input.clone()
                        grad_mask = torch.rand(grad.shape).cuda() > self.gt
                        spike_input[grad_mask == 1] = image_ones[grad_mask == 1] - spike_input[grad_mask == 1]

                    modi_pixel = torch.sum((spike_input - pre_spike_input), -1)
                    modi_pixel /= self.time_window
                    input_pre = input.clone().detach()
                    input += modi_pixel
                    input -= self.c * (input_pre - org_input)

                    input[input > 1] = 1
                    input[input < 0] = 0

                    # sample new input
                    spike_input = self.sample_input(input)
                    set_thresh_pernultimate(thresh_origin_value)
                    output = model(spike_input)
                    _, pred = torch.max(output.cpu().data, 1)

                    tmp_diff = torch.mean((org_input - input) ** 2).item()

                    if pred == j and torch.sum(output == torch.max(output)) == 1 and tmp_diff < self.max_diff:
                        self.succ += 1
                        succ_flag = True
                        self.avg_diff += tmp_diff

                    if iter == self.num_iter - 1 or succ_flag or tmp_diff > self.max_diff:
                        if not succ_flag:
                            self.fail += 1

                        if (i * 9 + j + 1) % 25 == 0 and self.succ != 0:
                            print(str(i + 1) + '\tsucc: ' + str(self.succ) + '\tfail: ' + str(self.fail) +
                                  '\twrong: ' + str(self.wrong) + '\tavg diff: ' + str(
                                self.avg_diff / (1.0 * self.succ)))

                        if (i * 9 + j + 1) % 25 == 0 and self.succ == 0:
                            print(str(i + 1) + '\tsucc: ' + str(self.succ) + '\tfail: ' + str(self.fail) +
                                  '\twrong: ' + str(self.wrong))

                        break


attack = Attack()

if attack_mode == 'untarget':
    attack.attack_untarget()

elif attack_mode == 'target':
    attack.attack_target()


torch.save({'succ': attack.succ, 'fail': attack.fail, 'wrong': attack.wrong,
            'ratio': (attack.succ * 1.0 / (attack.succ + attack.fail)),
            'avg_diff': (attack.avg_diff * 1.0 / attack.succ)},
           './ckpt/result/' + attack_mode + '_' + str(thresh_attack_value) + '.pth.tar')