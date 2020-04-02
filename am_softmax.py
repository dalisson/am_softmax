import torch
import torch.nn as nn
import torch.nn.functional as F


class AMSoftmax(nn.Module):
    '''
    The am softmax as seen on https://arxiv.org/pdf/1801.05599.pdf,
    in_features: size of the embedding, eg. 512
    n_classes: number of classes on the classification task
    s: s parameter of loss, standard = 30.
    m: m parameter of loss, standard = 0.4, best between 0.35 and 0.4 according to paper.
    *inputs:(embeddings, labels)
            (tensor (batch_size X embedding_size), tensor(batch_size))
    output: tensor shaped (batch_size X n_classes)
            these are the softmax softmax logits which can then passed
            through a NLLoss layer.

    '''
    def __init__(self, in_features, n_classes, s=30, m=0.4):
        super(AMSoftmax, self).__init__()
        self.linear = nn.Linear(in_features, n_classes, bias=False)
        self.s = s
        self.m = m

    def forward(self, *inputs):
        x_vector = F.normalize(inputs[0], p=2, dim=-1)
        self.linear.weight.data = F.normalize(self.linear.weight.data, p=2, dim=-1)
        logits = self.linear(x_vector)
        scaled_logits = (logits - self.m)*self.s
        am_softmax_logits = scaled_logits - self._am_logsumexp(logits)
        return F.nll_loss(am_softmax_logits, inputs[1])

    def _am_logsumexp(self, logits):
        '''
        logsumexp designed for am_softmax, the computation is numerically stable

        '''
        max_x = torch.max(logits, dim=-1)[0].unsqueeze(-1)
        term1 = (self.s*(logits - (max_x + self.m))).exp()
        term2 = (self.s * (logits - max_x)).exp().sum(-1).unsqueeze(-1) \
                - (self.s * (logits - max_x)).exp()
        return self.s*max_x + (term2 + term1).log()
