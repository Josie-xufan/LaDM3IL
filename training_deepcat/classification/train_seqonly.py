import os
from os import system
import os.path
from os.path import isdir
import sys
sys.path.append(os.path.abspath('..'))

import torch
from torch.utils.data import DataLoader
from Dataset import DatasetMultimodal
from Dataset_singleSentence import Dataset_singleSentence

from bert.dataset import WordVocab
from bert.model import BERT

from FusionModel import FusionModel
from gene.gene_encoder import get_gene_token, save_gene_token, load_saved_token
from gene.gene_encoder import get_gene_encoder
from loss_confidence import partial_loss

import argparse
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from datetime import datetime
import nni
import logging
import random
import gc
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score
import time
import warnings
warnings.filterwarnings("ignore")

#%%
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False   

def binary_accuracy(predict, label):
    rounded_predict = torch.round(predict)
    correct = (rounded_predict == label).float()
    accuracy = correct.sum() / len(correct)
    return accuracy

def train(bert_model, gene_model, model, gene_optimizer, bert_optimizer, optimizer, dataset_loader, train_phase, device, start_upd_prot):
    epoch_loss, epoch_acc = 0., 0.
    total_len = 0

    total_real = torch.empty(0,dtype=int)
    total_pred = torch.empty(0)
    total_ID = torch.empty(0,dtype=int)
    total_pred.to(device)

    for i, data in enumerate(dataset_loader):
        if(train_phase):
            if(args.finetune):
                bert_model.train()
            else:
                bert_model.eval()            
            model.train()        
        else:
            bert_model.eval()
            model.eval()
        
        item = data["item"]
        ID = data['ID']
        label = data['classification_label'].to(device)
        encoding = data['bert_input']
        segment_info = data['segment_label']
        
        bert_output = bert_model(encoding.to(device), segment_info.to(device))
        pooler_output = bert_output[:,0,:]
        Y_prob, Y_hat, features, prototypes_scores = model(pooler_output, None, start_upd_prot=start_upd_prot, train_phase=train_phase)
        
        #gd
        if(train_phase):
            if start_upd_prot:
                loss_fun.confidence_update(temp_un_conf=prototypes_scores,
                                        patch_index=item, 
                                        device=device)
                loss = loss_fun(Y_prob, item)
                Y_prob = Y_prob.squeeze(axis=1)
            else:
                Y_prob = Y_prob.squeeze(axis=1)
                loss = crit(Y_prob, label.float())

            optimizer.zero_grad()
            if(args.finetune):
                bert_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if(args.finetune):
                bert_optimizer.step()
        else:
            Y_prob = Y_prob.squeeze(axis=1)
            loss = crit(Y_prob, label.float())
        acc = binary_accuracy(Y_prob, label)

        epoch_loss += (loss.item()) * len(label)
        epoch_acc += acc * len(label)
        total_len += len(label)
        
        total_real = torch.cat([total_real.to(device),label],dim=0)
        total_pred = torch.cat([total_pred.to(device),Y_prob],dim=0)
        total_ID = torch.cat([total_ID,ID],dim=0)
    auc = roc_auc_score(total_real.detach().cpu().numpy(), total_pred.detach().cpu().numpy())
    return epoch_loss/total_len, epoch_acc/total_len, auc, total_real.detach().cpu().numpy(), total_pred.detach().cpu().numpy(), total_ID.numpy()

# early stop
def stop_check(loss,stop_criterion,stop_criterion_window):
    w = loss[-stop_criterion_window:]
    if((w[0]-w[-1])/w[0] < stop_criterion):
        return 1
    else:
        return 0

# plot loss&auc
def plot(Loss_list, Accuracy_list, outputname ,name, x):

    x1 = range(0, len(Loss_list))
    x2 = range(0, len(Accuracy_list))
    y1 = Loss_list
    y2 = Accuracy_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1)

    plt.title(name)
    plt.ylabel('loss')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2)

    plt.xlabel(x)
    plt.ylabel('auc')
    plt.savefig(outputname)
    plt.close()

#%%
if __name__ == '__main__':
    # Parameter setting#####################################################################################################################
    print("Parameter setting", "*"*100)
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vocab_path", 
                        required=False, 
                        type=str, 
                        default = os.path.abspath('../../pretrained/vocab_3mer.pkl'), 
                        help="built vocab model path with bert-vocab"
                    )
    parser.add_argument("-g", "--gene_token", 
                        required=False, 
                        type=str, 
                        default = os.path.abspath('../../dataset/deepcat/train_val_test/gene.csv'), 
                        help="built vocab model path with bert-vocab"
                    )
    parser.add_argument("-c", "--train_dataset", 
                        required=False, 
                        type=str, 
                        default=os.path.abspath('../../dataset/deepcat/train_val_test/train.tsv'), 
                        help="train dataset"
                    )
    parser.add_argument("-d", "--valid_dataset", 
                        required=False, 
                        type=str, 
                        default=os.path.abspath('../../dataset/deepcat/train_val_test/val.tsv'), 
                        help="valid dataset"
                    )
    parser.add_argument("-t", "--test_dataset", 
                        type=str, 
                        default=os.path.abspath('../../dataset/deepcat/train_val_test/test.tsv'), 
                        help="test dateset"
                    )
    parser.add_argument("--bert_model", 
                        type=str, 
                        default=os.path.abspath('../../pretrained/ab_3mer_len79.ep28'), 
                        help="bert model"
                    )
    parser.add_argument("-o", "--output_path", 
                        required=False, 
                        type=str, 
                        default=os.path.abspath('../../result/deepcat'), 
                        help="ex)output/bert.model"
                    )
    parser.add_argument('--updprot_threshold', type=float, default=0.9, help='threshold to update prototypes')
    parser.add_argument('--prot_start', type=int, default=15, help = 'Start Prototype Updating')
    parser.add_argument('--proto_ema', type=float, default=0.9633221964035881, help='momentum for computing the momving average of prototypes')
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="number of batch_size")
    parser.add_argument('--reuse_ckpt_dir', type=str, default=None, help='The path to reuse the pre-trained model')


    parser.add_argument("-s", "--seq_len", type=int, default=55, help="maximum sequence len")
    parser.add_argument("--prob", type=float, default=0.0, help="prob")
    parser.add_argument('--soft_ema_range', type=str, default='0.95,0.8', help='soften target updating coefficient (phi)')
    parser.add_argument("-e", "--epochs", type=int, default=100, help="min epochs")
    parser.add_argument("--lr_a", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--lr_b", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--lr_c", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--with_cuda", type=bool,  default=True, help="")
    parser.add_argument("--class_name", type=str, default='Classification', help="class name")
    parser.add_argument("--finetune", type=int, default=1, help="finetune bert")
    parser.add_argument("--chain", type=int, default=1, help="the number of chain")
    parser.add_argument("--seed", type=int, default=48, help="default seed")
    parser.add_argument("--NNI_Search", type=int, default=0, help="NNI Search")
    
    args = parser.parse_args() #args=["--reuse_ckpt_dir", "../../pretrained/deepcat_classification/max_auc_model.pth"]
    class_name = args.class_name

    if args.NNI_Search:
        print('Use NNI Search!')
        RCV_CONFIG = nni.get_next_parameter()
        args.batch_size = RCV_CONFIG['batch_size']
        args.updprot_threshold = RCV_CONFIG['updprot_threshold']
        args.prot_start = RCV_CONFIG['prot_start']
        args.proto_ema = RCV_CONFIG['proto_ema']
        seed = RCV_CONFIG['seed']
        path = os.path.join(args.output_path, class_name,'bs_{}_updprot_{}_protstart_{}_protoema_{}_seed_{}'.format(args.batch_size, args.updprot_threshold, args.prot_start, args.proto_ema, seed))
    else:
        loca=time.strftime('%Y-%m-%d-%H-%M-%S')
        path = os.path.join(args.output_path,class_name+"_"+str(loca))
        seed = args.seed
    args.soft_ema_range = [float(item) for item in args.soft_ema_range.split(',')]
    ########################################################################################################################################

    # Data and Model Loading################################################################################################################
    print("Data and Model Loading", "*"*100)
    train_dataset = args.train_dataset
    valid_dataset = args.valid_dataset
    test_dataset = args.test_dataset
    setup_seed(seed)

    print("Loading Vocab")
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))
    if(args.chain==1):
        Dataset = Dataset_singleSentence

    print("Loading Train Dataset")
    train_dataset = DatasetMultimodal(train_dataset, 
                                vocab, 
                                seq_len=args.seq_len,
                                on_memory=True,
                                prob = args.prob,
                                class_name = class_name)
    valid_dataset = DatasetMultimodal(valid_dataset, 
                                vocab, 
                                seq_len=args.seq_len,
                                on_memory=True,
                                prob = args.prob,
                                class_name = class_name)
    test_dataset = DatasetMultimodal(test_dataset, 
                                vocab, 
                                seq_len=args.seq_len,
                                on_memory=True,
                                prob = args.prob,
                                class_name = class_name)
    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=32, shuffle=True, drop_last=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=32)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=32)

    print("Creating Model")
    device = torch.device("cuda:" + str(2) if torch.cuda.is_available() else 'cpu')
    bert_model = torch.load(args.bert_model)
    bert_model = bert_model.to(device)

    networkOption = dict()
    networkOption['fusion_type']='pofusion'
    networkOption['skip']= True
    networkOption['use_bilinear']= True
    networkOption['input1_gate']= True
    networkOption['input2_gate']= True
    networkOption['input1_dim']= 512
    networkOption['input2_dim']= 24
    networkOption['input1_scale']= 4 
    networkOption['input2_scale']= 1
    networkOption['mmhid']=64
    networkOption['dropout_rate']=0.25
    networkOption['label_dim'] = 1
    networkOption['activation'] ='Sigmoid'
    networkOption['updprot'] = args.updprot_threshold
    networkOption['proto_ema'] = args.proto_ema
    model = FusionModel(networkOption)
    model = model.to(device)

    if args.reuse_ckpt_dir:
        checkpoint = torch.load(args.reuse_ckpt_dir, map_location=device)
        bert_model.load_state_dict(checkpoint['bert_model'])
        model.load_state_dict(checkpoint["fc_model"])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_c)
    bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=args.lr_b)
    crit = torch.nn.BCELoss()
    train_patch_labels = torch.tensor(np.asarray([int(float(i[3])) for i in train_dataset.lines])).int()
    confidence = torch.zeros(train_dataset.corpus_lines, 2)
    confidence[torch.arange(train_dataset.corpus_lines).long(), train_patch_labels.long()] = 1
    confidence = confidence.to(device)
    loss_fun = partial_loss(confidence) 
    ########################################################################################################################################

    #Start Training#########################################################################################################################
    print("Start Training", "*"*100)
    index = 0
    train_loss_total,val_loss_total = [],[]
    stop_check_list = []
    epoch = 0
    epochs_min = 10

    min_loss = 100
    max_auc = 0
    min_loss_auc,min_loss_acc = 0,0
    max_auc_auc,max_auc_acc, max_auc_f1 = 0,0,0
    last_epoch_auc,last_epoch_acc = 0,0

    epoch_train_loss_list = []
    epoch_train_auc_list = []
    epoch_valid_loss_list = []
    epoch_valid_auc_list = []
    epoch_test_loss_list = []
    epoch_test_auc_list = []

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(path)

    if not args.reuse_ckpt_dir:
        while(True):
            start_time = datetime.now()
            gc.collect()
            torch.cuda.empty_cache()
            epoch_train_loss, epoch_train_acc,epoch_train_auc, epoch_train_real, epoch_train_pred, epoch_train_ID = train(bert_model, 
                                                                                                                        None, 
                                                                                                                        model, 
                                                                                                                        None,
                                                                                                                        bert_optimizer,
                                                                                                                        optimizer,
                                                                                                                        dataset_loader = train_data_loader, 
                                                                                                                        train_phase = True, 
                                                                                                                        device = device,
                                                                                                                        start_upd_prot=index>args.prot_start
                                                                                                                )
            if index>args.prot_start:
                print("noisy label") 
                loss_fun.set_conf_ema_m(epoch, args)

            epoch_train_loss_list.append(epoch_train_loss)
            epoch_train_auc_list.append(epoch_train_auc)
            index += 1
            print("EPOCH %d_train loss:%f accuracy:%f auc:%f" % (index, epoch_train_loss, epoch_train_acc, epoch_train_auc))
            train_loss_total.append(epoch_train_loss)
            stop_criterion = 0.001
            stop_criterion_window = 10

            with torch.no_grad():
                epoch_valid_loss, epoch_valid_acc,epoch_valid_auc,epoch_valid_real, epoch_valid_pred, epoch_valid_ID = train(bert_model, 
                                                                                                                            None, 
                                                                                                                            model, 
                                                                                                                            None,
                                                                                                                            bert_optimizer,
                                                                                                                            optimizer,
                                                                                                                            dataset_loader = valid_data_loader,
                                                                                                                            train_phase=False,
                                                                                                                            device = device,
                                                                                                                            start_upd_prot=index>args.prot_start
                                                                                                                            )
                epoch_valid_loss_list.append(epoch_valid_loss)
                epoch_valid_auc_list.append(epoch_valid_auc)
                print("EPOCH %d_valid loss:%f accuracy:%f auc:%f" % (index, epoch_valid_loss, epoch_valid_acc, epoch_valid_auc))
                val_loss_total.append(epoch_valid_loss)

                epoch_test_loss, epoch_test_acc,epoch_test_auc,epoch_test_real, epoch_test_pred, epoch_test_ID = train(bert_model, 
                                                                                                                        None, 
                                                                                                                        model, 
                                                                                                                        None,
                                                                                                                        bert_optimizer,
                                                                                                                        optimizer,
                                                                                                                        dataset_loader = test_data_loader,
                                                                                                                        train_phase=False,
                                                                                                                        device = device,
                                                                                                                        start_upd_prot=index>args.prot_start
                                                                                                                    )
                epoch_test_loss_list.append(epoch_test_loss)
                epoch_test_auc_list.append(epoch_test_auc)          
                print("EPOCH %d_test loss:%f accuracy:%f auc:%f" % (index, epoch_test_loss, epoch_test_acc,epoch_test_auc))
                end_time = datetime.now()
                gc.collect()
                torch.cuda.empty_cache()
                print(f"time: {end_time - start_time}")  
                print("\n")
                
                # save model and results with minimum valid_loss
                if(min_loss > epoch_valid_loss):
                    min_loss = epoch_valid_loss
                    min_loss_auc = epoch_test_auc
                    min_loss_acc = epoch_test_acc
                    model_output = os.path.join(path,'min_loss_model.pth')
                    state = {
                        'bert_model':bert_model.state_dict(),
                        'fc_model':model.state_dict()
                    }
                    torch.save(state, model_output)
                    data = {
                        'ID':epoch_test_ID,
                        'real':epoch_test_real.tolist(),
                        'pred':epoch_test_pred.tolist()
                    }
                    df = pd.DataFrame(data)
                    df.to_csv(os.path.join(path,'min_loss_result.csv'),index=None)

                # save model and results with max valid_auc
                if(max_auc<epoch_valid_auc):
                    max_auc = epoch_valid_auc
                    max_auc_auc = epoch_test_auc
                    max_auc_acc = epoch_test_acc
                    model_output = os.path.join(path,'max_auc_model.pth')
                    state = {
                        'bert_model':bert_model.state_dict(),
                        'fc_model':model.state_dict()
                    }
                    torch.save(state, model_output)
                    data = {
                        'ID':epoch_test_ID,
                        'real':epoch_test_real.tolist(),
                        'pred':epoch_test_pred.tolist()
                    }
                    df = pd.DataFrame(data)
                    df.to_csv(os.path.join(path,'max_auc_result.csv'),index=None)
                    fpr, tpr, thresholds = metrics.roc_curve(df["real"], df["pred"])
                    max_auc_f1 = f1_score(df["real"], df["pred"]>=0.5, average='macro')

            if(epoch>args.epochs):
                break
            epoch += 1
    ########################################################################################################################################
    else:
    #Start Inference#########################################################################################################################
        index = 100
        max_auc = 0
        max_auc_auc,max_auc_acc,max_auc_f1 = 0,0,0

        with torch.no_grad():
            epoch_valid_loss, epoch_valid_acc,epoch_valid_auc,epoch_valid_real, epoch_valid_pred, epoch_valid_ID = train(bert_model, 
                                                                                                                        None, 
                                                                                                                        model, 
                                                                                                                        None,
                                                                                                                        bert_optimizer,
                                                                                                                        optimizer,
                                                                                                                        dataset_loader = valid_data_loader,
                                                                                                                        train_phase=False,
                                                                                                                        device = device,
                                                                                                                        start_upd_prot=index>args.prot_start
                                                                                                                        )
            epoch_valid_loss_list.append(epoch_valid_loss)
            epoch_valid_auc_list.append(epoch_valid_auc)
            print("EPOCH %d_valid loss:%f accuracy:%f auc:%f" % (index, epoch_valid_loss, epoch_valid_acc, epoch_valid_auc))
            val_loss_total.append(epoch_valid_loss)

            epoch_test_loss, epoch_test_acc,epoch_test_auc,epoch_test_real, epoch_test_pred, epoch_test_ID = train(bert_model, 
                                                                                                                    None, 
                                                                                                                    model, 
                                                                                                                    None,
                                                                                                                    bert_optimizer,
                                                                                                                    optimizer,
                                                                                                                    dataset_loader = test_data_loader,
                                                                                                                    train_phase=False,
                                                                                                                    device = device,
                                                                                                                    start_upd_prot=index>args.prot_start
                                                                                                                )
            epoch_test_loss_list.append(epoch_test_loss)
            epoch_test_auc_list.append(epoch_test_auc)          
            print("EPOCH %d_test loss:%f accuracy:%f auc:%f" % (index, epoch_test_loss, epoch_test_acc,epoch_test_auc))
            end_time = datetime.now()
            gc.collect()
            torch.cuda.empty_cache()

            # save result with max valid_auc
            max_auc = epoch_valid_auc
            max_auc_auc = epoch_test_auc
            max_auc_acc = epoch_test_acc

            data = {
                'ID':epoch_test_ID,
                'real':epoch_test_real.tolist(),
                'pred':epoch_test_pred.tolist()
            }
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(path,'max_auc_result.csv'),index=None)
            fpr, tpr, thresholds = metrics.roc_curve(df["real"], df["pred"])
            max_auc_f1 = f1_score(df["real"], df["pred"]>=0.5 , average='macro')          

    #Save Results###########################################################################################################################
    auc_csv = pd.DataFrame(columns=['lr_b','lr_c'])
    auc_csv['max_auc_auc'] = [max_auc_auc]
    auc_csv['max_auc_acc'] = [max_auc_acc.item()]
    auc_csv['max_auc_f1'] = [max_auc_f1]

    # auc_csv['min_loss_auc'] = [min_loss_auc]
    # auc_csv['min_loss_acc'] = [min_loss_acc.item()]

    # auc_csv['last_epoch_auc'] = [last_epoch_auc]
    # auc_csv['last_epoch_acc'] = [last_epoch_acc.item()]

    auc_csv['bert_model'] = args.bert_model
    auc_csv['finetune'] = args.finetune
    auc_csv['prob'] = args.prob
    auc_csv['seq_len'] = args.seq_len
    auc_csv['batch_size'] = args.batch_size
    auc_csv['lr_b'] = args.lr_b
    auc_csv['lr_c'] = args.lr_c

    auc_csv.to_csv(os.path.join(path,'parameters.csv'),index=None)
    nni.report_final_result(max_auc_auc)
    print('auc:', max_auc_auc)
    print('acc:', max_auc_acc.item())
    print('f1:', max_auc_f1)
    ########################################################################################################################################
# %%
9