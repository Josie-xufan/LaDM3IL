import torch
import torch.nn as nn

class gene_extractor(nn.Module):
    """
    gene_extractor model.
    """

    def __init__(self, hyperpareameters):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """
        super(gene_extractor, self).__init__()
        
        self.hyperpareameters = hyperpareameters
        self.embedding = nn.Embedding(hyperpareameters['gene_width'], hyperpareameters['gene_embed'], max_norm=True)
        self.dropout = nn.Dropout(p=hyperpareameters['dropout'])

    def forward(self, x):
        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)
        x = torch.reshape(x, (-1, self.hyperpareameters['gene_embed']))
        x = self.dropout(x)
        return x
