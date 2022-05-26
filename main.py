#!/usr/bin/env python3

import json
import models
import utils
import argparse,random,logging,numpy,os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader,RandomSampler,BatchSampler
# mycode
from torch.nn.utils import clip_grad_norm_
# from torch.nn.utils import clip_grad_norm
from time import time
from tqdm import tqdm
import traceback
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sumeval.metrics.rouge import RougeCalculator


logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')
parser = argparse.ArgumentParser(description='extractive summary')
# model
parser.add_argument('-save_dir',type=str,default='checkpoints/')
parser.add_argument('-embed_dim',type=int,default=100)
parser.add_argument('-embed_num',type=int,default=100)
parser.add_argument('-pos_dim',type=int,default=50)
parser.add_argument('-pos_num',type=int,default=100)
parser.add_argument('-seg_num',type=int,default=10)
parser.add_argument('-kernel_num',type=int,default=100)
parser.add_argument('-kernel_sizes',type=str,default='3,4,5')
parser.add_argument('-model',type=str,default='RNN_RNN')
parser.add_argument('-hidden_size',type=int,default=200)
# train
parser.add_argument('-lr',type=float,default=1e-4)
parser.add_argument('-batch_size',type=int,default=32)
parser.add_argument('-epochs',type=int,default=10)
parser.add_argument('-seed',type=int,default=1)
parser.add_argument('-train_dir',type=str,default='mydata/train.json')
parser.add_argument('-val_dir',type=str,default='mydata/valid.json')
parser.add_argument('-embedding',type=str,default='mydata/embedding.npz')
parser.add_argument('-word2id',type=str,default='mydata/word2id.json')
parser.add_argument('-report_every',type=int,default=500)
parser.add_argument('-seq_trunc',type=int,default=50)
parser.add_argument('-max_norm',type=float,default=1.0)
# test
parser.add_argument('-load_dir',type=str,default='checkpoints/RNN_RNN_seed_1.pt')
parser.add_argument('-test_dir',type=str,default='mydata/test.json')
parser.add_argument('-ref',type=str,default='outputs/ref')
parser.add_argument('-hyp',type=str,default='outputs/hyp')
parser.add_argument('-filename',type=str,default='x.txt') # deprecated, TextFile to be summarized
parser.add_argument('-topk',type=int,default=15)
# device
parser.add_argument('-device',type=int)
# option
parser.add_argument('-test',action='store_true')
parser.add_argument('-debug',action='store_true')
parser.add_argument('-predict',action='store_true')
args = parser.parse_args()
use_gpu = args.device is not None

if torch.cuda.is_available() and not use_gpu:
    print("WARNING: You have a CUDA device, should run with -device 0")

# set cuda device and seed
if use_gpu:
    torch.cuda.set_device(args.device)
    print("using gpu")
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
numpy.random.seed(args.seed) 

def cal_rouge(rouge:RougeCalculator,sum,ref):
    rouge_1 = rouge.rouge_n(
                summary=sum,
                references=ref,
                n=1)

    rouge_2 = rouge.rouge_n(
                summary=sum,
                references=ref,
                n=2)

    rouge_l = rouge.rouge_l(
                summary=sum,
                references=ref)


    return rouge_1, rouge_2, rouge_l
    
def eval(net,vocab,data_iter,criterion):
    net.eval()
    total_loss = 0
    batch_num = 0
    doc_num=0
    rouge = RougeCalculator(stopwords=False, lang="zh")
    rouge_1, rouge_2, rouge_l=0,0,0
    for batch in data_iter:
        features,targets,summaries,doc_lens = vocab.make_features(batch)#将文本转为id
        doc_num+=len(batch)
        features,targets = Variable(features), Variable(targets.float())
        if use_gpu:
            features = features.cuda()
            targets = targets.cuda()
        probs = net(features,doc_lens)

        # calc rouge
        l=0
        for doc_id,doc_len in enumerate(doc_lens):
            this_doc_probs=probs[l:l+doc_len]
            # print(f"this_doc_probs {this_doc_probs.cpu().detach().numpy()}")
            # print(np.where(np.array(this_doc_probs.cpu().detach().numpy())>0.5))
            for pred_sum_idx in np.where(np.array(this_doc_probs.cpu().detach().numpy())>0.5)[0]:
                # print(f"features.cpu().data.numpy()[l+pred_sum_idx] {features.cpu().data.numpy()[l+pred_sum_idx]}")
                # print(l+pred_sum_idx)
                pred_sum=' '.join(vocab.feature2words(features.cpu().data.numpy()[l+pred_sum_idx]))
                # print(pred_sum)
                ground_truth=summaries[doc_id]
                r1,r2,rL=cal_rouge(rouge,ground_truth,pred_sum)
                rouge_1+=r1
                rouge_2+=r2
                rouge_l+=rL
            l+=doc_len

        loss = criterion(probs,targets)
        # mycode
        total_loss += loss.data
        # total_loss += loss.data[0]
        batch_num += 1
    loss = total_loss / batch_num
    rouge_1/=doc_num
    rouge_2/=doc_num
    rouge_l/=doc_num
    print(f"rouge_1, rouge_2, rouge_l {rouge_1},{rouge_2},{rouge_l}")
    net.train()
    return loss,probs,targets

def train():
    logging.info('Loading vocab,train and val dataset.Wait a second,please')
    
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)

    with open(args.train_dir) as f:
        examples = [json.loads(line) for line in f]
    train_dataset = utils.Dataset(examples)

    with open(args.val_dir) as f:
        examples = [json.loads(line) for line in f]
    val_dataset = utils.Dataset(examples)

    # update args
    args.embed_num = embed.size(0)
    args.embed_dim = embed.size(1)
    args.kernel_sizes = [int(ks) for ks in args.kernel_sizes.split(',')]
    # build model
    net = getattr(models,args.model)(args,embed)
    if use_gpu:
        net.cuda()
    # load dataset
    train_sampler=RandomSampler(train_dataset,num_samples=int(0.1*len(train_dataset)))
    train_iter = DataLoader(dataset=train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler)

    val_iter = DataLoader(dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False)
    # loss function
    criterion = nn.BCELoss()
    # model info
    print(net)
    params = sum(p.numel() for p in list(net.parameters())) / 1e6
    print('#Params: %.1fM' % (params))
    
    min_loss = float('inf')
    cur_loss = float('inf')
    optimizer = torch.optim.Adam(net.parameters(),lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    net.train()
    
    for epoch in range(1,args.epochs+1):
        t1 = time() 
        for i,batch in enumerate(train_iter):
            features,targets,summaries,doc_lens = vocab.make_features(batch)
            # print(f'batch[0] {batch[0]}')
            # print(f"doc {batch['doc']}")
            # print(f"labels {batch['labels']}")
            # print(f"summaries {summaries}")
            # print(f"features {features.shape} {features}")
            # print(f"targets {targets.shape} {targets}")
            # print(f"doc_lens sum {sum(doc_lens)} {doc_lens}")
            features,targets = Variable(features), Variable(targets.float())
            if use_gpu:
                features = features.cuda()
                targets = targets.cuda()
            probs = net(features,doc_lens)
            # print(f"probs {probs}")
            loss = criterion(probs,targets)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(net.parameters(), args.max_norm)
            # clip_grad_norm(net.parameters(), args.max_norm)
            optimizer.step()
            if args.debug:
                print('Batch ID:%d Loss:%f' %(i,loss.data[0]))
                continue
            if i % args.report_every == 0:
                cur_loss,_probs,_targets = eval(net,vocab,val_iter,criterion)
                if cur_loss < min_loss:
                    min_loss = cur_loss
                    best_path = net.save()
                logging.info('Epoch: %2d Min_Val_Loss: %f Cur_Val_Loss: %f'
                        % (epoch,min_loss,cur_loss))
                # logging.info(f'{probs},{targets}')
            # assert 0
        scheduler.step(cur_loss)
        logging.info(f"lr {optimizer.param_groups[0]['lr']}")
        t2 = time()
        logging.info('epoch Cost:%f h'%((t2-t1)/3600))

def test():
     
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)

    with open(args.test_dir) as f:
        examples = [json.loads(line) for line in f]
    test_dataset = utils.Dataset(examples)

    test_iter = DataLoader(dataset=test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    if use_gpu:
        checkpoint = torch.load(args.load_dir)
    else:
        checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)

    # checkpoint['args']['device'] saves the device used as train time
    # if at test time, we are using a CPU, we must override device to None
    if not use_gpu:
        checkpoint['args'].device = None
    net = getattr(models,checkpoint['args'].model)(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])
    if use_gpu:
        net.cuda()
    net.eval()
    
    doc_num = len(test_dataset)
    time_cost = 0
    file_id = 1
    for batch in tqdm(test_iter):
        features,tgt,summaries,doc_lens = vocab.make_features(batch)
        t1 = time()
        if use_gpu:
            probs = net(Variable(features).cuda(), doc_lens)
        else:
            probs = net(Variable(features), doc_lens)
        t2 = time()
        time_cost += t2 - t1
        start = 0
        for doc_id,doc_len in enumerate(doc_lens):
            stop = start + doc_len
            if doc_len==1:
                # print(probs.cpu())
                # print(probs.cpu().data)
                # print(probs.cpu().data.numpy())
                # print()
                # print(probs.cpu().data.numpy()[start:stop])
                prob = probs
            else:
                prob = probs[start:stop]
            topk = min(args.topk,doc_len)
            topk_indices = prob.topk(topk)[1].cpu().data.numpy()
            if doc_len!=1:
                topk_indices.sort()
            doc = batch['doc'][doc_id].split('\n')[:doc_len]
            if doc_len==1:
                hyp = [doc[topk_indices]]
            else:
                hyp = [doc[index] for index in topk_indices]
            ref = summaries[doc_id]
            with open(os.path.join(args.ref,str(file_id)+'.txt'), 'w') as f:
                f.write(ref)
            with open(os.path.join(args.hyp,str(file_id)+'.txt'), 'w') as f:
                f.write('\n'.join(hyp))
            start = stop
            file_id = file_id + 1


    print('Speed: %.2f docs / s' % (doc_num / time_cost))


def predict(examples):
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)
    pred_dataset = utils.Dataset(examples)

    pred_iter = DataLoader(dataset=pred_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    if use_gpu:
        checkpoint = torch.load(args.load_dir)
    else:
        checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)

    # checkpoint['args']['device'] saves the device used as train time
    # if at test time, we are using a CPU, we must override device to None
    if not use_gpu:
        checkpoint['args'].device = None
    net = getattr(models,checkpoint['args'].model)(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])
    if use_gpu:
        net.cuda()
    net.eval()
    
    doc_num = len(pred_dataset)
    time_cost = 0
    file_id = 1
    for batch in tqdm(pred_iter):
        print(batch)
        features, doc_lens = vocab.make_predict_features(batch)
        t1 = time()
        if use_gpu:
            probs = net(Variable(features).cuda(), doc_lens)
        else:
            probs = net(Variable(features), doc_lens)
        t2 = time()
        time_cost += t2 - t1
        start = 0
        for doc_id,doc_len in enumerate(doc_lens):
            stop = start + doc_len
            print(probs)
            prob = probs[start:stop]
            topk = min(args.topk,doc_len)
            topk_indices = prob.topk(topk)[1].cpu().data.numpy()
            topk_indices.sort()
            doc = batch[doc_id].split('. ')[:doc_len]
            hyp = [doc[index] for index in topk_indices]
            with open(os.path.join(args.hyp,str(file_id)+'.txt'), 'w') as f:
                f.write('. '.join(hyp))
            start = stop
            file_id = file_id + 1
    print('Speed: %.2f docs / s' % (doc_num / time_cost))

if __name__=='__main__':
    if args.test:
        test()
    elif args.predict:
        with open(args.filename) as file:
            bod = [file.read()]
        predict(bod)
    else:
        train()
