import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
import os

# torch.set_default_tensor_type('torch.cuda.HalfTensor')
#%%
class FusionModel(nn.Module):
    def __init__(self, options):
        super(FusionModel, self).__init__()
        self.fusion = define_bifusion(fusion_type=options['fusion_type'], 
                                      skip=options['skip'], 
                                      use_bilinear=options['use_bilinear'], 
                                      gate1=options['input1_gate'], 
                                      gate2=options['input2_gate'], 
                                      dim1=options['input1_dim'], 
                                      dim2=options['input2_dim'], 
                                      scale_dim1=options['input1_scale'], 
                                      scale_dim2=options['input2_scale'], 
                                      mmhid=options['mmhid'], 
                                      dropout_rate=options['dropout_rate'])
        self.classifier = nn.Sequential(nn.Linear(options['mmhid'], options['label_dim']))
        self.activation = define_act_layer(act_type=options['activation'])

        self.register_buffer("prototypes_embed", torch.zeros(options['label_dim']+1, options['mmhid']))
        self.updprot = options['updprot']
        self.proto_ema = options['proto_ema']

    def forward(self, feature_tensor_one, feature_tensor_two, start_upd_prot=False, train_phase=False):
        # memoryleft = os.popen("free -k").readlines()
        # print(f"before feature fusion memoryused",memoryleft[1].split()[2], " KB")

        features = self.fusion(feature_tensor_one, feature_tensor_two)
        Y_prob = self.classifier(features)
        if self.activation is not None:
            Y_prob = self.activation(Y_prob)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        if not train_phase:
            return Y_prob, Y_hat, features, None

        # memoryleft = os.popen("free -k").readlines()
        # print(f"after feature fusion memoryused",memoryleft[1].split()[2], " KB")

        # noisy label ############################################################################
        prototypes_embed = self.prototypes_embed.clone().detach() # pseudo_num_class x embed_dim
        prototypes_score = torch.mm(features, prototypes_embed.t())
        prototypes_scores = torch.softmax(prototypes_score, dim=1) # patch_num x pseudo_num_class
        if start_upd_prot:
            for feat_encoder, pseudo_label, val in zip(features, Y_hat, torch.max(torch.concat([1-Y_prob, Y_prob], axis=1), dim=1)[0]):
                if val>self.updprot: 
                    self.prototypes_embed[pseudo_label.long()] = self.prototypes_embed[pseudo_label.long()]*self.proto_ema + (1-self.proto_ema)*feat_encoder.detach()
            self.prototypes_embed = F.normalize(self.prototypes_embed, p=2, dim=1)

        # memoryleft = os.popen("free -k").readlines()
        # print(f"after prototypes update memoryused",memoryleft[1].split()[2], " KB","\n")

        return Y_prob, Y_hat, features, prototypes_scores
        ##########################################################################################


    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False


def define_bifusion(fusion_type, skip=True, use_bilinear=True, 
                    gate1=True, gate2=True, gated_fusion=True,
                    dim1=32, dim2=32, 
                    scale_dim1=1, scale_dim2=1, 
                    mmhid=64, dropout_rate=0.25):
    fusion = None
    if fusion_type == 'pofusion':
        fusion = BilinearFusion(skip=skip, use_bilinear=use_bilinear, 
                                gate1=gate1, gate2=gate2, gated_fusion=gated_fusion,
                                dim1=dim1, dim2=dim2, 
                                scale_dim1=scale_dim1, scale_dim2=scale_dim2, 
                                mmhid=mmhid, dropout_rate=dropout_rate)
    else:
        raise NotImplementedError('fusion type [%s] is not found' % fusion_type)
    return fusion

class BilinearFusion(nn.Module):
    def __init__(self, skip=True, use_bilinear=True, 
                 gate1=True, gate2=True, gated_fusion=True,
                 dim1=32, dim2=32, 
                 scale_dim1=1, scale_dim2=1, 
                 mmhid=64, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        self.gated_fusion = gated_fusion

        dim1_og, dim2_og = dim1, dim2 
        dim1, dim2 = int(dim1/scale_dim1), int(dim2/scale_dim2)

        skip_dim = dim1+dim2+2 if skip else 0
        
        # A sequential container. Modules will be added to it in the order they are passed in the constructor. 
        # Alternatively, an ordered dict of modules can also be passed in.
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        # a bilinear transformation to the incoming data y = x1Ax2+b
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(mmhid+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        # self.encoder3 = nn.Sequential(nn.Linear(dim1+dim2, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder3 = nn.Sequential(nn.Linear(dim1, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        init_max_weights(self)

    def forward(self, vec1, vec2):
        device = torch.device("cuda:" + str(2) if torch.cuda.is_available() else 'cpu')

        if vec2!=None:
            ### Gated Multimodal Units
            if self.gate1:
                h1 = self.linear_h1(vec1) #512->128
                z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
                o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
            else:
                o1 = self.linear_h1(vec1)

            if self.gate2:
                h2 = self.linear_h2(vec2)
                z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
                o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
            else:
                o2 = self.linear_h2(vec2)

            ### Fusion
            if self.gated_fusion:
                o1 = torch.cat((o1, torch.FloatTensor(o1.shape[0], 1).fill_(1).to(device)), 1)
                o2 = torch.cat((o2, torch.FloatTensor(o2.shape[0], 1).fill_(1).to(device)), 1)
                o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
                out = self.post_fusion_dropout(o12)
                out = self.encoder1(out)
                if self.skip: 
                    out = torch.cat((out, o1, o2), 1)
                out = self.encoder2(out)
                
            else:
                out = torch.cat((o1, o2), 1)
                out = self.encoder3(out)
            return out
        else:
            o1 = self.linear_h1(vec1)
            out = self.encoder3(o1)
            return out            

#%%
def define_act_layer(act_type='Tanh'):
    if act_type == 'Tanh':
        act_layer = nn.Tanh()
    elif act_type == 'ReLU':
        act_layer = nn.ReLU()
    elif act_type == 'Sigmoid':
        act_layer = nn.Sigmoid()
    elif act_type == 'LSM':
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == "none":
        act_layer = None
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer

def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)

def dfs_unfreeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_unfreeze(child)

def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()

#%%
# AUXILIARY METHODS
def calculate_classification_error(Y_hat, Y):
    Y = Y.float()
    error = 1. - Y_hat.eq(Y).cpu().float().mean().data

    return error

def calculate_loss(Y_prob, Y):
    Y = Y.float()
    Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5) # Clamp all elements in input into the range [ min, max ] and return a resulting tensor:
    neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

    return neg_log_likelihood

def performance_statistics(Y_prob, Y_hat, Y):
    Y = Y.float()
    Y_true = Y.data.cpu().detach().numpy()
    Y_true = Y_true > 0.5
    Y_true = Y_true.astype('int32')     
    Y_prob = Y_prob.data.cpu().detach().numpy()    
    
    right_prediction_count = torch.sum(Y_hat.eq(Y))
    right_prediction_count = right_prediction_count.data.cpu().detach().numpy()
    # print('Y_true:{} , Y_pred:{}, is_equal:{}'.format(Y_true, Y_hat, Y_hat.eq(Y)))

    return Y_true, Y_prob, right_prediction_count

def calculate_ROCAUC(Y_ture_list, Y_prob_list):
    Y_ture = np.array(Y_ture_list)
    Y_prob = np.array(Y_prob_list)
    # print('-'*30)
    # print('y-true:{}, y-prob:{}'.format(Y_ture.shape, Y_prob.shape))
    # print('-'*30)
    ROCAUC = roc_auc_score(Y_ture, Y_prob) if len(np.unique(Y_ture)) > 1 else 0.0
    return ROCAUC

def calculate_confusion_matrix(Y_true_list, Y_hat_list):
    Y_true = np.array(Y_true_list)
    Y_hat = np.array(Y_hat_list)
    TN, FP, FN, TP = confusion_matrix(Y_true, Y_hat).ravel() if len(np.unique(Y_true)) > 1 else 0.0
    return TN, FP, FN, TP

def calculate_precision_score(Y_true_list, Y_hat_list):
    Y_true = np.array(Y_true_list)
    Y_hat = np.array(Y_hat_list)
    precision = precision_score(Y_true, Y_hat) if len(np.unique(Y_true)) > 1 else 0.0
    return precision

def calculate_recall_score(Y_true_list, Y_hat_list):
    Y_true = np.array(Y_true_list)
    Y_hat = np.array(Y_hat_list)
    recall = recall_score(Y_true, Y_hat) if len(np.unique(Y_true)) > 1 else 0.0
    return recall

def calculate_f1_score(Y_true_list, Y_hat_list):
    Y_true = np.array(Y_true_list)
    Y_hat = np.array(Y_hat_list)
    f1 = f1_score(Y_true, Y_hat) if len(np.unique(Y_true)) > 1 else 0.0
    return f1
