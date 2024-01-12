import os
from os import system
import os.path
from os.path import isdir
import sys
sys.path.append('..')

from torch.utils.tensorboard import SummaryWriter
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
import tqdm as tqdm
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import metrics
import numpy as np
import pandas as pd
from datetime import datetime
import nni
import logging
import random
import gc
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import stats
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
    total_item = torch.empty(0, dtype=int)
    total_fre = torch.empty(0)
    total_pred.to(device)
    sample_prediction = {}

    for i, data in enumerate(dataset_loader):
        if(train_phase):
            if(args.finetune):
                bert_model.train()
            else:
                bert_model.eval()            
            gene_model.train()    
            model.train()        
        else:
            bert_model.eval()
            gene_model.eval() 
            model.eval()

        item = data["item"]
        ID = data['ID']
        fre = data['fre']
        label = data['classification_label'].to(device)
        encoding = data['bert_input']
        segment_info = data['segment_label']
            
        gene_info_dict = dict()
        gene_info_dict["TRA_v_gene"] = data["TRA_v_gene"]
        gene_info_dict["TRA_j_gene"] = data["TRA_j_gene"]
        gene_df = pd.DataFrame(gene_info_dict)
        gene_info_input = {c:gene_df[c].values for c in gene_info_dict.keys()}

        # sequence feature
        bert_output = bert_model(encoding.to(device), segment_info.to(device))
        pooler_output = bert_output[:,0,:]
        # gene feature 
        tokenized_feature = gene_model.tokenize(gene_info_input)
        tokenized_feature = {key: torch.tensor(value).to(device) for key, value in tokenized_feature.items()} 
        gene_output = gene_model(tokenized_feature)
        
        Y_prob, Y_hat, features, prototypes_scores = model(pooler_output, gene_output, start_upd_prot=start_upd_prot, train_phase=train_phase)

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
            gene_optimizer.zero_grad()
            if(args.finetune):
                bert_optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            gene_optimizer.step()
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
        total_item = torch.cat([total_item, item], dim=0)
        total_fre = torch.cat([total_fre, fre], dim=0)
    auc = roc_auc_score(total_real.detach().cpu().numpy(), total_pred.detach().cpu().numpy())

    # compute sample auc and acc#############################################################################################
    for s, spred, slabel, sfre in zip(total_ID.numpy(), total_pred.detach().cpu().numpy(), total_real.detach().cpu().numpy(), total_fre.cpu().numpy()):
        if s not in sample_prediction:
            sample_prediction[s] = []
        sample_prediction[s].append([spred, sfre, slabel])
    ids = np.asarray(list(sample_prediction.keys()))
    labels = []
    for s in sample_prediction:
        assert len(list(set([_itemk[2] for _itemk in sample_prediction[s]])))==1
        labels.append(list(set([_itemk[2] for _itemk in sample_prediction[s]]))[0])
        if start_upd_prot:
            sample_prediction[s]=sum([_itemi*_itemj if abs(_itemi-0.5)>=0.1 else 0 for _itemi, _itemj, _ in sample_prediction[s]])
        else:
            sample_prediction[s]=sum([_itemi*_itemj for _itemi, _itemj, _ in sample_prediction[s]])
    if start_upd_prot:
        sample_prediction=[sample_prediction[_item] for _item in sample_prediction] #.item()
    else:
        sample_prediction=[sample_prediction[_item].item() for _item in sample_prediction] #
    sample_prediction=np.asarray(sample_prediction)
    # print(np.min(sample_prediction), np.max(sample_prediction), np.max(sample_prediction)-np.min(sample_prediction))
    if np.max(sample_prediction)>0:
        sample_prediction=(sample_prediction-np.min(sample_prediction))/(np.max(sample_prediction)-np.min(sample_prediction))
        labels=np.asarray(labels)
        fpr, tpr, thresholds = metrics.roc_curve(labels, sample_prediction)
        sample_auc=metrics.auc(fpr, tpr)
        threshold = thresholds[np.argmax(tpr - fpr)]
        sample_f1 = f1_score(labels, sample_prediction>threshold , average='macro')
        sample_acc = accuracy_score(labels, sample_prediction>threshold)
    elif np.max(sample_prediction)==0:
        labels=np.asarray(labels)
        fpr, tpr, thresholds = metrics.roc_curve(labels, sample_prediction)
        sample_auc=metrics.auc(fpr, tpr)
        threshold = thresholds[np.argmax(tpr - fpr)]
        sample_f1 = f1_score(labels, sample_prediction>threshold , average='macro')
        sample_acc = accuracy_score(labels, sample_prediction>threshold)
    return epoch_loss/total_len, epoch_acc/total_len, auc, labels, sample_prediction, ids, total_item.numpy(), sample_auc, sample_f1, sample_acc, total_real.detach().cpu().numpy(), total_pred.detach().cpu().numpy(), total_ID.numpy()

# early stop
def stop_check(loss,stop_criterion,stop_criterion_window):
    w = loss[-stop_criterion_window:]
    # print(w)
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

def plot1_probdist(seq_pred, seq, inputs, cmv_associated, cur_dataset="test"):
    print(f"current dataset {cur_dataset}")
    cur_dfpred = seq_pred.copy()
    cur_dfpred["pred_norm"] = (cur_dfpred["pred"]-cur_dfpred["pred"].min())/(cur_dfpred["pred"].max()-cur_dfpred["pred"].min())
    cur_dfpred["seq"] = seq
    cur_dfpred_cmvassociated = cur_dfpred[cur_dfpred["seq"].isin(cmv_associated)]
    cur_dfpred_cmvassociated["type"] = ["cmv associated"]*cur_dfpred_cmvassociated.shape[0]
    cur_dfpred_others = cur_dfpred[~cur_dfpred["seq"].isin(cmv_associated)]
    cur_dfpred_others["type"] = ["others"]*cur_dfpred_others.shape[0]
    cur_dfpred_cmvassociated["seq_label"] = 1
    cur_dfpred_others["seq_label"] = 0

    print(stats.levene(cur_dfpred_others["pred_norm"], cur_dfpred_cmvassociated["pred_norm"]))
    print(stats.ttest_ind(cur_dfpred_others["pred_norm"], cur_dfpred_cmvassociated["pred_norm"], equal_var=False))

    cur_dfpred_all = pd.concat([cur_dfpred_cmvassociated, cur_dfpred_others])
    fpr, tpr, thresholds = metrics.roc_curve(cur_dfpred_all["seq_label"], cur_dfpred_all["pred"])
    seq_auc=metrics.auc(fpr, tpr)
    return seq_auc

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
                        default = os.path.abspath('../../dataset/cmv/train_val_test/gene.csv'), 
                        help="built vocab model path with bert-vocab"
                    )
    parser.add_argument("-c", "--train_dataset", 
                        required=False, 
                        type=str, 
                        default=os.path.abspath('../../dataset/cmv/train_val_test/train.tsv'), 
                        help="train dataset"
                    )
    parser.add_argument("-d", "--valid_dataset", 
                        required=False, 
                        type=str, 
                        default=os.path.abspath('../../dataset/cmv/train_val_test/val.tsv'), 
                        help="valid dataset"
                    )
    parser.add_argument("-t", "--test_dataset", 
                        type=str, 
                        default=os.path.abspath('../../dataset/cmv/train_val_test/test.tsv'), 
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
                        default=os.path.abspath('../../result/cmv'), 
                        help="ex)output/bert.model"
                    )
    parser.add_argument('--updprot_threshold', type=float, default=0.8, help='threshold to update prototypes')
    parser.add_argument('--prot_start', type=int, default=15, help = 'Start Prototype Updating')
    parser.add_argument('--proto_ema', type=float, default=0.8104110009077465, help='momentum for computing the momving average of prototypes')
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="number of batch_size")
    parser.add_argument('--reuse_ckpt_dir', type=str, default=None, help='The path to reuse the pre-trained model')


    parser.add_argument("-s", "--seq_len", type=int, default=30, help="maximum sequence len")
    parser.add_argument("--prob", type=float, default=0.0, help="prob")
    parser.add_argument('--soft_ema_range', type=str, default='0.95,0.8', help='soften target updating coefficient (phi)')
    parser.add_argument("-e", "--epochs", type=int, default=80, help="min epochs")
    parser.add_argument("--lr_a", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--lr_b", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--lr_c", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--with_cuda", type=bool,  default=True, help="")
    parser.add_argument("--class_name", type=str, default='Classification', help="class name")
    parser.add_argument("--finetune", type=int, default=1, help="finetune bert")
    parser.add_argument("--chain", type=int, default=1, help="the number of chain")
    parser.add_argument("--seed", type=int, default=57, help="default seed")
    parser.add_argument("--NNI_Search", type=int, default=0, help="NNI Search")
    args = parser.parse_args() #args=["--reuse_ckpt_dir", "../../pretrained/cmv_classification/max_auc_model.pth"]
    class_name = args.class_name

    # NNI search
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

    print("build gene tokens")
    dataframe = pd.read_csv(os.path.abspath(args.gene_token), dtype={'ID':int, 'TRA_v_gene': str, 'TRA_j_gene': str, 'TRB_v_gene': str, 'TRB_j_gene': str})
    print(dataframe.head(5))
    gene_tokenizers = get_gene_token(dataframe = dataframe)
    if not os.path.exists(os.path.join(args.output_path, "gene_tokenizers")):
        save_gene_token(tokenizers=gene_tokenizers, output_folder=args.output_path)
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

    device = torch.device("cuda:" + str(1) if torch.cuda.is_available() else 'cpu')
    bert_model = torch.load(args.bert_model)
    bert_model = bert_model.to(device)
    gene_model = get_gene_encoder(gene_tokenizer = gene_tokenizers)
    gene_model = gene_model.to(device)

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
        gene_model.load_state_dict(checkpoint['gene_model'])
        bert_model.load_state_dict(checkpoint['bert_model'])
        model.load_state_dict(checkpoint["fc_model"])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_c)
    gene_optimizer = torch.optim.Adam(gene_model.parameters(), lr=args.lr_a)
    bert_optimizer = torch.optim.Adam(bert_model.parameters(), lr=args.lr_b)

    crit = torch.nn.BCELoss()
    train_patch_labels = torch.tensor(np.asarray([int(i[3]) for i in train_dataset.lines])).int()
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
    min_loss_auc,min_loss_acc,min_loss_f1 = 0,0,0
    max_auc_auc,max_auc_acc,max_auc_f1 = 0,0,0
    last_epoch_auc,last_epoch_acc = 0,0
    max_testauc, max_testacc, max_testf1 = 0,0,0

    epoch_train_loss_list = []
    epoch_train_auc_list = []
    epoch_valid_loss_list = []
    epoch_valid_auc_list = []
    epoch_test_loss_list = []
    epoch_test_auc_list = []

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    batchwriter = SummaryWriter(os.path.join(path, "tensorboard"))


    if not args.reuse_ckpt_dir:
        while(True):
            start_time = datetime.now()
            gc.collect()
            torch.cuda.empty_cache()
            epoch_train_loss, epoch_train_acc,epoch_train_auc, epoch_train_real, epoch_train_pred, epoch_train_ID, epoch_train_item, epoch_train_sampleauc, epoch_train_samplef1, epoch_train_sampleacc, epoch_train_seqreal, epoch_train_seqpred, epoch_train_seqid = train(bert_model, 
                                                                                                                        gene_model, 
                                                                                                                        model, 
                                                                                                                        gene_optimizer,
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
            print("EPOCH %d_train loss:%f accuracy:%f auc:%f sampleauc:%f samplef1:%f sampleacc:%f" % (index, epoch_train_loss, epoch_train_acc, epoch_train_auc, epoch_train_sampleauc, epoch_train_samplef1, epoch_train_sampleacc))
            
            train_loss_total.append(epoch_train_loss)
            stop_criterion = 0.001
            stop_criterion_window = 10

            with torch.no_grad():
                epoch_valid_loss, epoch_valid_acc,epoch_valid_auc,epoch_valid_real, epoch_valid_pred, epoch_valid_ID,_, epoch_val_sampleauc, epoch_val_samplef1, epoch_val_sampleacc, epoch_val_seqreal, epoch_val_seqpred, epoch_val_seqid = train(bert_model, 
                                                                                                                            gene_model, 
                                                                                                                            model, 
                                                                                                                            gene_optimizer,
                                                                                                                            bert_optimizer,
                                                                                                                            optimizer,
                                                                                                                            dataset_loader = valid_data_loader,
                                                                                                                            train_phase=False,
                                                                                                                            device = device,
                                                                                                                            start_upd_prot=index>args.prot_start
                                                                                                                            )
                epoch_valid_loss_list.append(epoch_valid_loss)
                epoch_valid_auc_list.append(epoch_valid_auc)
                print("EPOCH %d_valid loss:%f accuracy:%f auc:%f sampleauc:%f samplef1:%f sampleacc:%f" % (index, epoch_valid_loss, epoch_valid_acc, epoch_valid_auc, epoch_val_sampleauc, epoch_val_samplef1, epoch_val_sampleacc))
                val_loss_total.append(epoch_valid_loss)

                epoch_test_loss, epoch_test_acc,epoch_test_auc,epoch_test_real, epoch_test_pred, epoch_test_ID,_, epoch_test_sampleauc, epoch_test_samplef1, epoch_test_sampleacc, epoch_test_seqreal, epoch_test_seqpred, epoch_test_seqid = train(bert_model, 
                                                                                                                        gene_model, 
                                                                                                                        model, 
                                                                                                                        gene_optimizer,
                                                                                                                        bert_optimizer,
                                                                                                                        optimizer,
                                                                                                                        dataset_loader = test_data_loader,
                                                                                                                        train_phase=False,
                                                                                                                        device = device,
                                                                                                                        start_upd_prot=index>args.prot_start
                                                                                                                    )
                epoch_test_loss_list.append(epoch_test_loss)
                epoch_test_auc_list.append(epoch_test_auc)          
                print("EPOCH %d_test loss:%f accuracy:%f auc:%f sampleauc:%f samplef1:%f sampleacc:%f" % (index, epoch_test_loss, epoch_test_acc,epoch_test_auc,epoch_test_sampleauc, epoch_test_samplef1, epoch_test_sampleacc))
                end_time = datetime.now()
                gc.collect()
                torch.cuda.empty_cache()
                print(f"time: {end_time - start_time}")  
                print("\n")
                
                # before noisy label, save confidence matrix and test sequence prediction
                if (index-1) == args.prot_start:
                    matrix_confidence = pd.DataFrame(loss_fun.confidence.cpu().numpy())
                    matrix_confidence.columns = ["label0", "label1"]
                    
                    real_item = {
                        'item': epoch_train_item,
                        'real': epoch_train_seqreal.tolist(),
                        'pred': epoch_train_seqpred.tolist()
                    }
                    real_item = pd.DataFrame(real_item)
                    real_item = real_item.sort_values(by="item")
                    real_item = real_item.reset_index(drop=True)

                    matrix_confidence["gt"] = real_item["real"]
                    matrix_confidence["pred"] = real_item["pred"]
                    matrix_confidence["item"] = real_item["item"]
                    matrix_confidence.to_csv(os.path.join(path, f"confidence_{index}.csv"))

                    testdata = {
                        'ID': epoch_test_seqid,
                        'real': epoch_test_seqreal.tolist(),
                        'pred': epoch_test_seqpred.tolist()
                    }
                    testdf = pd.DataFrame(testdata)
                    testdf.to_csv(os.path.join(path, f'testprediction_at{index}.csv'), index=None)

                # save model and results with minimum valid_loss
                if(min_loss > epoch_valid_loss):
                    min_loss = epoch_valid_loss
                    min_loss_auc = epoch_test_sampleauc
                    min_loss_acc = epoch_test_sampleacc
                    min_loss_f1 = epoch_test_samplef1
                    
                    model_output = os.path.join(path,'min_loss_model.pth')
                    state = {
                        'bert_model':bert_model.state_dict(),
                        'fc_model':model.state_dict(),
                        'gene_model': gene_model.state_dict()
                    }
                    torch.save(state, model_output)
                    data = {
                        'ID':epoch_test_ID,
                        'real':epoch_test_real.tolist(),
                        'pred':epoch_test_pred.tolist()
                    }
                    df = pd.DataFrame(data)
                    df.to_csv(os.path.join(path,'min_loss_result.csv'),index=None)

                    testdata = {
                        'ID': epoch_test_seqid,
                        'real': epoch_test_seqreal.tolist(),
                        'pred': epoch_test_seqpred.tolist()
                    }
                    testdf = pd.DataFrame(testdata)
                    testdf.to_csv(os.path.join(path, f'testprediction_minloss.csv'), index=None)

                    traindata = {
                        'ID': epoch_train_seqid,
                        'item': epoch_train_item, 
                        'real': epoch_train_seqreal.tolist(),
                        'pred': epoch_train_seqpred.tolist()
                    }
                    traindf = pd.DataFrame(traindata)
                    traindf.to_csv(os.path.join(path, f'trainprediction_minloss.csv'), index=None)

                    valdata = {
                        'ID': epoch_val_seqid,
                        'real': epoch_val_seqreal.tolist(),
                        'pred': epoch_val_seqpred.tolist()
                    }
                    valdf = pd.DataFrame(valdata)
                    valdf.to_csv(os.path.join(path, f'valprediction_minloss.csv'), index=None)

                # save model and results with max valid_auc
                if(max_auc<epoch_val_sampleauc):
                    max_auc = epoch_val_sampleauc
                    max_auc_auc = epoch_test_sampleauc
                    max_auc_acc = epoch_test_sampleacc
                    max_auc_f1 = epoch_test_samplef1
                    model_output = os.path.join(path,'max_auc_model.pth')
                    state = {
                        'bert_model':bert_model.state_dict(),
                        'fc_model':model.state_dict(),
                        'gene_model': gene_model.state_dict()
                    }
                    torch.save(state, model_output)

                    data = {
                        'ID':epoch_test_ID,
                        'real':epoch_test_real.tolist(),
                        'pred':epoch_test_pred.tolist()
                    }
                    df = pd.DataFrame(data)
                    df.to_csv(os.path.join(path,'max_auc_result.csv'),index=None)

                    testdata = {
                        'ID': epoch_test_seqid,
                        'real': epoch_test_seqreal.tolist(),
                        'pred': epoch_test_seqpred.tolist()
                    }
                    testdf = pd.DataFrame(testdata)
                    testdf.to_csv(os.path.join(path, f'testprediction_maxauc.csv'), index=None)

                    traindata = {
                        'ID': epoch_train_seqid,
                        'item': epoch_train_item, 
                        'real': epoch_train_seqreal.tolist(),
                        'pred': epoch_train_seqpred.tolist()
                    }
                    traindf = pd.DataFrame(traindata)
                    traindf.to_csv(os.path.join(path, f'trainprediction_maxauc.csv'), index=None)

                    valdata = {
                        'ID': epoch_val_seqid,
                        'real': epoch_val_seqreal.tolist(),
                        'pred': epoch_val_seqpred.tolist()
                    }
                    valdf = pd.DataFrame(valdata)
                    valdf.to_csv(os.path.join(path, f'valprediction_maxauc.csv'), index=None)

                    if (index-1)>args.prot_start:
                        matrix_confidence = pd.DataFrame(loss_fun.confidence.cpu().numpy())
                        matrix_confidence.columns = ["label0", "label1"]
                        
                        real_item = {
                            'item': epoch_train_item,
                            'real': epoch_train_seqreal.tolist(),
                            'pred': epoch_train_seqpred.tolist()
                        }
                        real_item = pd.DataFrame(real_item)
                        real_item = real_item.sort_values(by="item")
                        real_item = real_item.reset_index(drop=True)

                        matrix_confidence["gt"] = real_item["real"]
                        matrix_confidence["pred"] = real_item["pred"]
                        matrix_confidence["item"] = real_item["item"]
                        matrix_confidence.to_csv(os.path.join(path, "confidence_maxauc.csv"))

                # save model and results with max test_auc
                if(max_testauc<epoch_test_sampleauc):
                    max_testauc = epoch_test_sampleauc
                    max_testacc = epoch_test_sampleacc
                    max_testf1 = epoch_test_samplef1
                    model_output = os.path.join(path,'max_testauc_model.pth')
                    state = {
                        'bert_model':bert_model.state_dict(),
                        'fc_model':model.state_dict(),
                        'gene_model': gene_model.state_dict()
                    }
                    torch.save(state, model_output)

                    data = {
                        'ID':epoch_test_ID,
                        'real':epoch_test_real.tolist(),
                        'pred':epoch_test_pred.tolist()
                    }
                    df = pd.DataFrame(data)
                    df.to_csv(os.path.join(path,'max_testauc_result.csv'),index=None)

                    testdata = {
                        'ID': epoch_test_seqid,
                        'real': epoch_test_seqreal.tolist(),
                        'pred': epoch_test_seqpred.tolist()
                    }
                    testdf = pd.DataFrame(testdata)
                    testdf.to_csv(os.path.join(path, f'testprediction_maxtestauc.csv'), index=None)

                    traindata = {
                        'ID': epoch_train_seqid,
                        'item': epoch_train_item, 
                        'real': epoch_train_seqreal.tolist(),
                        'pred': epoch_train_seqpred.tolist()
                    }
                    traindf = pd.DataFrame(traindata)
                    traindf.to_csv(os.path.join(path, f'trainprediction_maxtestauc.csv'), index=None)

                    valdata = {
                        'ID': epoch_val_seqid,
                        'real': epoch_val_seqreal.tolist(),
                        'pred': epoch_val_seqpred.tolist()
                    }
                    valdf = pd.DataFrame(valdata)
                    valdf.to_csv(os.path.join(path, f'valprediction_maxtestauc.csv'), index=None)

                    if (index-1)>args.prot_start:
                        matrix_confidence = pd.DataFrame(loss_fun.confidence.cpu().numpy())
                        matrix_confidence.columns = ["label0", "label1"]
                        
                        real_item = {
                            'item': epoch_train_item,
                            'real': epoch_train_seqreal.tolist(),
                            'pred': epoch_train_seqpred.tolist()
                        }
                        real_item = pd.DataFrame(real_item)
                        real_item = real_item.sort_values(by="item")
                        real_item = real_item.reset_index(drop=True)

                        matrix_confidence["gt"] = real_item["real"]
                        matrix_confidence["pred"] = real_item["pred"]
                        matrix_confidence["item"] = real_item["item"]

                        matrix_confidence.to_csv(os.path.join(path, "confidence_maxtestauc.csv"))

            batchwriter.add_scalar("EpochLoss/train", epoch_train_loss, index)
            batchwriter.add_scalar("EpochAUC/train", epoch_train_sampleauc, index)
            batchwriter.add_scalar("EpochAUC/val", epoch_val_sampleauc, index)
            batchwriter.add_scalar("EpochAUC/test", epoch_test_sampleauc, index)

            if(epoch>args.epochs):
                break
            epoch += 1
    ########################################################################################################################################
            
    #Start Inference#########################################################################################################################
    else:
        index = 80
        max_auc = 0
        max_auc_auc,max_auc_acc,max_auc_f1 = 0,0,0
        with torch.no_grad():
            epoch_valid_loss, epoch_valid_acc,epoch_valid_auc,epoch_valid_real, epoch_valid_pred, epoch_valid_ID,_, epoch_val_sampleauc, epoch_val_samplef1, epoch_val_sampleacc, epoch_val_seqreal, epoch_val_seqpred, epoch_val_seqid = train(bert_model, 
                                                                                                                        gene_model, 
                                                                                                                        model, 
                                                                                                                        gene_optimizer,
                                                                                                                        bert_optimizer,
                                                                                                                        optimizer,
                                                                                                                        dataset_loader = valid_data_loader,
                                                                                                                        train_phase=False,
                                                                                                                        device = device,
                                                                                                                        start_upd_prot=index>args.prot_start
                                                                                                                        )
            epoch_valid_loss_list.append(epoch_valid_loss)
            epoch_valid_auc_list.append(epoch_valid_auc)
            print("EPOCH %d_valid loss:%f accuracy:%f auc:%f sampleauc:%f samplef1:%f sampleacc:%f" % (index, epoch_valid_loss, epoch_valid_acc, epoch_valid_auc, epoch_val_sampleauc, epoch_val_samplef1, epoch_val_sampleacc))
            val_loss_total.append(epoch_valid_loss)

            epoch_test_loss, epoch_test_acc,epoch_test_auc,epoch_test_real, epoch_test_pred, epoch_test_ID,_, epoch_test_sampleauc, epoch_test_samplef1, epoch_test_sampleacc, epoch_test_seqreal, epoch_test_seqpred, epoch_test_seqid = train(bert_model, 
                                                                                                                    gene_model, 
                                                                                                                    model, 
                                                                                                                    gene_optimizer,
                                                                                                                    bert_optimizer,
                                                                                                                    optimizer,
                                                                                                                    dataset_loader = test_data_loader,
                                                                                                                    train_phase=False,
                                                                                                                    device = device,
                                                                                                                    start_upd_prot=index>args.prot_start
                                                                                                                )
            epoch_test_loss_list.append(epoch_test_loss)
            epoch_test_auc_list.append(epoch_test_auc)          
            print("EPOCH %d_test loss:%f accuracy:%f auc:%f sampleauc:%f samplef1:%f sampleacc:%f" % (index, epoch_test_loss, epoch_test_acc,epoch_test_auc,epoch_test_sampleauc, epoch_test_samplef1, epoch_test_sampleacc))
            end_time = datetime.now()
            gc.collect()
            torch.cuda.empty_cache()

            # save result with max valid_auc
            max_auc = epoch_val_sampleauc
            max_auc_auc = epoch_test_sampleauc
            max_auc_acc = epoch_test_sampleacc
            max_auc_f1 = epoch_test_samplef1

            data = {
                'ID':epoch_test_ID,
                'real':epoch_test_real.tolist(),
                'pred':epoch_test_pred.tolist()
            }
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(path,'max_auc_result.csv'),index=None)

            testdata = {
                'ID': epoch_test_seqid,
                'real': epoch_test_seqreal.tolist(),
                'pred': epoch_test_seqpred.tolist()
            }
            testdf = pd.DataFrame(testdata)
            testdf.to_csv(os.path.join(path, f'testprediction_maxauc.csv'), index=None)

            valdata = {
                'ID': epoch_val_seqid,
                'real': epoch_val_seqreal.tolist(),
                'pred': epoch_val_seqpred.tolist()
            }
            valdf = pd.DataFrame(valdata)
            valdf.to_csv(os.path.join(path, f'valprediction_maxauc.csv'), index=None)
    ########################################################################################################################################
            
    #Save Results###########################################################################################################################
    auc_csv = pd.DataFrame(columns=['lr_b','lr_c'])
    auc_csv['max_auc_auc'] = [max_auc_auc]
    auc_csv['max_auc_acc'] = [max_auc_acc.item()]
    auc_csv['max_auc_f1'] = [max_auc_f1.item()]

    # auc_csv['min_loss_auc'] = [min_loss_auc]
    # auc_csv['min_loss_acc'] = [min_loss_acc.item()]
    # auc_csv['min_loss_f1'] = [min_loss_f1.item()]
    # auc_csv['last_epoch_auc'] = [last_epoch_auc]
    # auc_csv['last_epoch_acc'] = [last_epoch_acc.item()]

    auc_csv['bert_model'] = args.bert_model
    auc_csv['finetune'] = args.finetune
    auc_csv['prob'] = args.prob
    auc_csv['seq_len'] = args.seq_len
    auc_csv['batch_size'] = args.batch_size
    auc_csv['updprot_threshold'] = args.updprot_threshold
    auc_csv['prot_start'] = args.prot_start
    auc_csv['proto_ema'] = args.proto_ema
    auc_csv['seed'] = args.seed
    auc_csv['lr_b'] = args.lr_b
    auc_csv['lr_c'] = args.lr_c
    auc_csv.to_csv(os.path.join(path,'parameters.csv'),index=None)

    print("val auc based")
    print('auc:',max_auc_auc)
    print('acc:',max_auc_acc.item())
    print('f1:',max_auc_f1.item())

    # print("test auc based")
    # print('auc:',max_testauc)
    # print('acc:',max_testacc.item())
    # print('f1:',max_testf1.item())
    # nni.report_final_result(max_auc_auc)
    ########################################################################################################################################

