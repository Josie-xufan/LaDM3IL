import os
import numpy as np
import torch
import torch.nn as nn
import sys
import pickle
sys.path.insert(0, os.path.abspath('../utility'))
from utility.tokenizer.text import Tokenizer
from .extractor import gene_extractor
#%%
gene_gene_embed_dim = {
    # 'TRB_v_gene': 16,
    # 'TRB_j_gene': 8,
    'TRA_v_gene': 16,
    'TRA_j_gene': 8
}

# gene_cols = ['TRB_v_gene','TRB_j_gene','TRA_v_gene','TRA_j_gene']
#%%
def get_gene_token(Tokenizer=Tokenizer, 
                   dataframe = None, 
                   gene_cols = ['TRA_v_gene','TRA_j_gene'] #'TRB_v_gene','TRB_j_gene',
                ):
    # VDJ gene
    gene_tokenizers = dict()    
    for col in gene_cols:
        print(dataframe[col])
        gene_tokenizers[col] = Tokenizer(filters='', lower=False, oov_token='UNK')
        gene_tokenizers[col].fit_on_texts(dataframe[col])
    return gene_tokenizers

def save_gene_token(tokenizers, output_folder):
        pickle_path = os.path.join(output_folder, 'gene_tokenizers')
        os.makedirs(pickle_path, exist_ok=True)
        pickle.dump(tokenizers, open(os.path.join(pickle_path,'tokenizers.pickle'), "wb"))
        
def load_saved_token(model_saved_folder):
    token_path = os.path.join(model_saved_folder, 'gene_tokenizers', 'tokenizers.pickle')
    gene_tokenizers = pickle.load(open(token_path, "rb"))
    return gene_tokenizers

class get_gene_encoder(nn.Module):
    """
    gene_extractor model.
    """

    def __init__(self, gene_tokenizer, gene_cols =  ['TRA_v_gene','TRA_j_gene'], gene_gene_embed_dim = gene_gene_embed_dim):
        """
        :param gene_cols: vocab_size of total words
        :param gene_tokenizer: BERT model hidden size
        :param gene_gene_embed_dim: numbers of Transformer blocks(layers)
        
        """
        super(get_gene_encoder, self).__init__()
        
        self.gene_cols = gene_cols
        self.gene_tokenizer = gene_tokenizer
        self.gene_encs = nn.ModuleDict()
        for v in gene_gene_embed_dim.keys():
            hp_gene={'gene_width': len(self.gene_tokenizer[v].word_index) + 1,
                     'gene_embed': gene_gene_embed_dim[v],
                     'dropout': 0.3
            }
            self.gene_encs[v] = gene_extractor(hp_gene)
    
    def tokenize(self, inputs):
        """ Apply tokenizers to inputs
    
        parameters
        -----------
        
        inputs: (dict) 
                dictionary of named inputs, {name: input data}. input data typically
                an array of strings.
                
        returns
        ----------
        tokenized: (dict)
                dictionary of tokenized outputs for the inputs that correspond 
                to a tokenizer key of self.tokenizers. 
        
        """
        tokenized = dict()
        for tok_name, tok in self.gene_tokenizer.items():
            tokenized[tok_name] = np.array(tok.texts_to_sequences(inputs[tok_name]))
        return tokenized
        
    def embedding_concatenate(self, inputs, selection=None):
        """ calculate latent space features
        
        parameters
        ----------
        
        inputs: (dict)
                dict of (np/tf) arrays of the inputs, keys should be the names
                of the inputs, and values the arrays.
                
        selection: (list, optional, default=None)
                only those inputs whose keys are in selection will be processed into
                the feature space. If None, the self.input_list will be used
                
        returns
        -------
        
        concatenated_feature: (np or tf array)
            The feature space vector of the inputs appearing in selection_list.
        
        """
        if selection is None:
            selection=self.gene_cols
        
        vals = list()
        for sel in selection:
            if sel in list(self.gene_encs.keys()):
                vals.append(self.gene_encs[sel](inputs[sel]))
            else:
                raise RuntimeError(f"unknown selection key passed: {sel}")
        
        if len(vals)>1:
            # --------------- for debugging --------------
            # for val in vals:
            #     print(val.size())
            z = torch.cat(vals, dim=1)
        else:
            z = vals[0]
        return z   

    def forward(self, inputs):
        # embedding the indexed sequence to sequence of vectors
        output = self.embedding_concatenate(inputs)
        return output
        


