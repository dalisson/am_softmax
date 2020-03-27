import torch
import torch.nn as nn
import torch.nn.functional as F

class AMSoftmax(nn.Module):
    '''
    the am softmax as seen on https://arxiv.org/pdf/1801.05599.pdf, expects a batch of embeddings as input.
    inputs : tensor shaped batch_sizeXembedding_size eg. 64X512, 64 embeddings of size 512 each
    outputs : softmax logits which can then be passed through a NLLoss layer
    
    '''
    def __init__(self, in_features, n_classes, s, m):
        super(AMSoftmax, self).__init__()
        self.linear = nn.Parameter(torch.empty(in_features, n_classes))
        self.s = s
        self.m = m
        self._init_parameter()

    def _init_parameter(self):
        nn.init.xavier_uniform_(self.linear, gain=nn.init.calculate_gain('relu'))

    def forward(self, *inputs):
        x_vector = F.normalize(inputs[0], p=2, dim=-1)
        kernel = F.normalize(self.linear, p=2, dim=0)
        logits = x_vector@kernel
        scaled_logits = (logits - self.m)*self.s
        return scaled_logits - self.am_logsumexp(logits)

    def am_logsumexp(self, logits):
        '''
        logsumexp desenhada para a am_softmax

        '''
        max_x = torch.max(logits, dim=-1)[0].unsqueeze(-1)
        term1 = (self.s*(logits - (max_x + self.m))).exp()
        term2 = -(self.s * (logits - max_x).exp()) \
                +((self.s * (logits - max_x)).exp().sum(-1)).unsqueeze(-1)

        return -self.s*max_x + (term2 + term1).log()
