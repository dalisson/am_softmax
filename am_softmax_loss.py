from torch.nn.functional import nll_loss
from am_softmax import AMSoftmax


class AMSoftmaxLoss(AMSoftmax):
    '''
        AM softmax loss, computes NLL_Loss using AM_Softmax logits.
        in_features: embedding_size
        n_classes: number of classes on classification task

        *inputs: (embeddings, labels)
                (tensor batch_sizeXembedding_size, tensor batch_size)
        output: NLL_Loss computed using AM_softmax logits
    '''

    def forward(self, *inputs):
        logits = super().forward(*inputs)
        return nll_loss(logits, inputs[1])
