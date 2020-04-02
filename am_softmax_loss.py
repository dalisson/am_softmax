import torch.nn as nn
from torch.nn.functional import nll_loss
from am_softmax import AMSoftmax


class AMSoftmaxLoss(nn.Module):
    '''
        AM softmax loss, computes NLL_Loss using AM_Softmax logits.
        in_features: embedding_size
        n_classes: number of classes on classification task

        *inputs: (embeddings, labels)
                (tensor batch_sizeXembedding_size, tensor batch_size)
        output: NLL_Loss computed using AM_softmax logits
    '''

    def __init__(self, in_features, n_classes, s=30, m=0.4):
        super(AMSoftmaxLoss, self).__init__()
        self.am_softmax = AMSoftmax(in_features, n_classes, s, m)


    def forward(self, *inputs):
        logits = self.am_softmax(inputs[0])
        return nll_loss(logits, inputs[1])
