from torch.utils.data import Dataset
from itertools import islice

import tqdm
import torch
import random

import sys
sys.path.append('../bert')
from dataset import BERTDataset_MLM


class DatasetMultimodal(BERTDataset_MLM):
    def __init__(self, 
                corpus_path, 
                vocab, 
                seq_len, 
                class_name, 
                encoding="utf-8", 
                corpus_lines=None, 
                on_memory=True,
                prob=0.10
            ):
        super().__init__(corpus_path, 
                         vocab, 
                         seq_len,
                         encoding, 
                         corpus_lines, 
                         on_memory,prob
                    )
        self.class_name = class_name

    def __getitem__(self, item):
        ID, TRA_cdr3_3Mer, TRB_cdr3_3Mer, epitope, TRA_cdr3, TRA_v_gene, TRA_j_gene, TRB_cdr3, TRB_v_gene, TRB_j_gene, fre = self.get_corpus_line(item)

        t1_random, t1_label = self.random_word(TRA_cdr3_3Mer)
        # t2_random, t2_label = self.random_word(TRB_cdr3_3Mer)
        # print('tokens:',t1)
        # print('t1_random:',t1_random)
        # print('t1_label:',t1_label)
        # print('\n')

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        # t2 = t2_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        # t2_label = t2_label + [self.vocab.pad_index]

        segment_label = ([1 for _ in range(len(t1))])[:self.seq_len] #([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1)[:self.seq_len] #(t1 + t2)[:self.seq_len]
        bert_label = (t1_label)[:self.seq_len] #(t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)
        
        # binary
        # if(epitope==self.class_name):
        #     label = 1
        # else:
        #     label = 0
        label = int(float(epitope))

        output = {"ID":int(ID),
                  "bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "classification_label":label,
                  "item": item,
                  "fre": float(fre)
                  }
        
        # print('using dataloader')
        output_tensor = {key: torch.tensor(value) for key, value in output.items()}
        output_tensor["TRA_v_gene"] = TRA_v_gene
        output_tensor["TRA_j_gene"] = TRA_j_gene
        output_tensor["TRB_v_gene"] = TRB_v_gene
        output_tensor["TRB_j_gene"] = TRB_j_gene
        return output_tensor

    def get_corpus_line(self, item):
        # 'ID','TRA_cdr3_3Mer','TRB_cdr3_3Mer','epitope', 'TRA_cdr3','TRA_v_gene','TRA_j_gene','TRB_cdr3','TRB_v_gene','TRB_j_gene'
        if self.on_memory:
            ID = self.lines[item][0]
            TRA_cdr3_3Mer = self.lines[item][1]
            TRB_cdr3_3Mer = self.lines[item][2]
            epitope = self.lines[item][3]
            TRA_cdr3 = self.lines[item][4]
            TRA_v_gene = self.lines[item][5]
            TRA_j_gene = self.lines[item][6]
            TRB_cdr3 = self.lines[item][7]
            TRB_v_gene = self.lines[item][8]
            TRB_j_gene = self.lines[item][9]
            fre = self.lines[item][10]
            return ID, TRA_cdr3_3Mer, TRB_cdr3_3Mer, epitope, TRA_cdr3, str(TRA_v_gene), str(TRA_j_gene), TRB_cdr3, str(TRB_v_gene), str(TRB_j_gene), fre
        else: 
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            ID, TRA_cdr3_3Mer, TRB_cdr3_3Mer, epitope, TRA_cdr3, TRA_v_gene, TRA_j_gene, TRB_cdr3, TRB_v_gene, TRB_j_gene = line[:-1].split("\t")
            return ID, TRA_cdr3_3Mer, TRB_cdr3_3Mer, epitope, TRA_cdr3, str(TRA_v_gene), str(TRA_j_gene), TRB_cdr3, str(TRB_v_gene), str(TRB_j_gene)


class Dataset(BERTDataset_MLM):
    def __init__(self, corpus_path, vocab, seq_len, class_name, encoding="utf-8", corpus_lines=None, on_memory=True,prob=0.10):
        super().__init__(corpus_path, vocab, seq_len,encoding, corpus_lines, on_memory,prob)
        self.class_name = class_name

    def __getitem__(self, item):
        t0,t1, t2, t3 = self.get_corpus_line(item)

        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)
        # print('tokens:',t1)
        # print('t1_random:',t1_random)
        # print('t1_label:',t1_label)
        # print('\n')

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)
        
        # binary
        if(t3==self.class_name):
            label = 1
        else:
            label = 0

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "classification_label":label,
                  "ID":int(t0)}
        output = {key: torch.tensor(value) for key, value in output.items()}
        # print(output)
        return output

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item][0], self.lines[item][1], self.lines[item][2], self.lines[item][3]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t0, t1, t2, t3 = line[:-1].split("\t")
            return t0, t1, t2, t3
