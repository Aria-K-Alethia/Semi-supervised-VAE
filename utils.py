import torch

def onehot(k):
    def encode(label):
        y = torch.zeros(k)
        if label < k:
            y[label] = 1
        return y
    return encode

def onehot_vector(label, number):
    # label: [#batch_size]
    # number: label number
    device = label.device
    out = torch.arange(number).to(device)
    out.unsqueeze_(0)
    batch_size = label.shape[0]
    out = out.repeat(batch_size, 1)
    label = label.unsqueeze(1)
    temp = (out != label)
    out[out == label] = 1
    out[temp] = 0
    return out

def unsupervised_label(x, nlabel):
    batch_size = x.shape[0]
    label = torch.cat([torch.ones(batch_size) * i for i in range(nlabel)])
    label = label.to(x.device)
    return label
