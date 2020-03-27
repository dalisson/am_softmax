**Pytorch implementation of additive margin softmax as seen on original paper**  
https://arxiv.org/pdf/1801.05599.pdf  
This is a loss function designed for embedding learning as it encourages intra-class distances to be small and inter-class distances to be larger, so the input to the layer should be a batchXembedding tensor and the output will be softmax logits which can then be passed to a NLL implementation.  
eg. input size 64X512 means 64 embeddings of size 512 each.
