#!/usr/bin/env python3

import argparse
import json
import numpy as np
from collections import OrderedDict
from glob import glob
from time import time
from multiprocessing import Pool,cpu_count
from itertools import chain
import gensim
from gensim.models import Word2Vec
from greedy_labeler import GreedyLabeler

def build_vocab(args):
    print('start building vocab')

    PAD_IDX = 0
    UNK_IDX = 1
    PAD_TOKEN = 'PAD_TOKEN'
    UNK_TOKEN = 'UNK_TOKEN'
    
    f = open(args.embed)
    embed_dim = int(next(f).split()[1])

    word2id = OrderedDict()
    
    word2id[PAD_TOKEN] = PAD_IDX
    word2id[UNK_TOKEN] = UNK_IDX
    
    embed_list = []
    # fill PAD and UNK vector
    embed_list.append([0 for _ in range(embed_dim)])
    embed_list.append([0 for _ in range(embed_dim)])
    
    # build Vocab
    for line in f:
        tokens = line.split()
        word = tokens[:-1*embed_dim][0]
        vector = [float(num) for num in tokens[-1*embed_dim:]]
        embed_list.append(vector)
        word2id[word] = len(word2id)
    f.close()
    embed = np.array(embed_list,dtype=np.float32)
    np.savez_compressed(file=args.vocab, embedding=embed)
    with open(args.word2id,'w',encoding='utf-8') as f:
        json.dump(word2id,f)




def worker(files):
    examples = []
    for f in files:
        parts = open(f,encoding='latin-1').read().split('\n\n')
        try:
            entities = { line.strip().split(':')[0]:line.strip().split(':')[1].lower() for line in parts[-1].split('\n')}
        except:
            continue
        sents,labels,summaries = [],[],[]
        # content
        for line in parts[1].strip().split('\n'):
            content, label = line.split('\t\t\t')
            tokens = content.strip().split()
            for i,token in enumerate(tokens):
                if token in entities:
                    tokens[i] = entities[token]
            label = '1' if label == '1' else '0'
            sents.append(' '.join(tokens))
            labels.append(label)
        # summary
        for line in parts[2].strip().split('\n'):
            tokens = line.strip().split()
            for i, token in enumerate(tokens):
                if token in entities:
                    tokens[i] = entities[token]
            line = ' '.join(tokens).replace('*','')
            summaries.append(line)
        ex = {'doc':'\n'.join(sents),'labels':'\n'.join(labels),'summaries':'\n'.join(summaries)}
        examples.append(ex)
    return examples

def build_dataset(args):
    t1 = time()
    
    print('start building dataset')
    if args.worker_num == 1 and cpu_count() > 1:
        print('[INFO] There are %d CPUs in your device, please increase -worker_num to speed up' % (cpu_count()))
        print("       It's a IO intensive application, so 2~10 may be a good choise")

    files = glob(args.source_dir)
    data_num = len(files)
    group_size = data_num // args.worker_num
    groups = []
    for i in range(args.worker_num):
        if i == args.worker_num - 1:
            groups.append(files[i*group_size : ])
        else:
            groups.append(files[i*group_size : (i+1)*group_size])
    p = Pool(processes=args.worker_num)
    multi_res = [p.apply_async(worker,(fs,)) for fs in groups]
    res = [res.get() for res in multi_res]
    
    with open(args.output_dir, 'w') as f:
        for row in chain(*res):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    t2 = time()
    print('Time Cost : %.1f seconds' % (t2 - t1))



def my_build_vocab(args):
    print('start building vocab')

    PAD_IDX = 0
    UNK_IDX = 1
    PAD_TOKEN = 'PAD_TOKEN'
    UNK_TOKEN = 'UNK_TOKEN'
    # f = open(args.embed)
    # embed_dim = int(next(f).split()[1])
    word2vec_model = Word2Vec.load("/content/drive/MyDrive/NLP/nlp_text_summarization/word2vec-Chinese/trained_models/lcsts.word2vec.model")
    embed_dim = len(word2vec_model.wv[word2vec_model.wv.index2word[0]])
    vocab_size=len(word2vec_model.wv.index2word)

    word2id = OrderedDict()
    
    word2id[PAD_TOKEN] = PAD_IDX
    word2id[UNK_TOKEN] = UNK_IDX
    
    embed_list = []
    # fill PAD and UNK vector
    embed_list.append([0]*embed_dim)
    embed_list.append([0]*embed_dim)
    
    # build Vocab
    for index in range(vocab_size):
        
        word = word2vec_model.wv.index2word[index]
        vector = word2vec_model.wv[word].tolist()
        embed_list.append(vector)
        word2id[word] = index+2
    # f.close()
    embed = np.array(embed_list,dtype=np.float32)
    # print(embed)
    np.savez_compressed(file=args.vocab, embedding=embed)
    with open(args.word2id,'w',encoding='utf-8') as f:
        f.write(json.dumps(word2id, ensure_ascii=False))



def my_worker(files_dir_path,mode):
    """
    将jieba切分好的数据按照train valid test进行整理，最终生成三个json
    Steps:
        1. 将带空格的数据（是个字符串）根据分隔符进行切分
        2. 对切分的子句进行切分，利用my_build_vocab生成的word2id模型将其转换为id
        3. 利用英文rouge计算三个得分
        --------------------------------
        1. 将带空格的数据（是个字符串）根据分隔符进行切分
        2. 对切分的子句进行切分，利用中文rouge计算三个得分
    """
    assert mode in ['train','valid','test']
    if files_dir_path[-1]!='/':
        files_dir_path+='/'

    suffix=['.src','.tgt']
    src_path=files_dir_path+mode+suffix[0]
    tgt_path=files_dir_path+mode+suffix[1]
    with open(src_path,'r',encoding='utf-8') as fr:
        src=fr.readlines()
    with open(tgt_path,'r',encoding='utf-8') as fr:
        tgt=fr.readlines()
    
    gl=GreedyLabeler()
    
    examples = []
    for pair in zip(src,tgt):
        sents=gl.sep_ref(pair[0].strip())
        summary=pair[1].strip()
        best,labels=gl.label(summary,sents)
        ex = {'doc':'\n'.join(sents).strip(),'labels':'\n'.join([str(idx) for idx in labels]),'summaries':summary.strip()}
        assert len(labels)>0 and ex['doc'][-1]!='\n'
        examples.append(ex)

    return examples


def my_build_dataset(args):
    for mode in ['valid','test','train']:
        t1 = time()
        
        print(f'start building dataset {mode}')        
        res=my_worker(args.source_dir,mode)
    
        with open(args.output_dir+mode+'.json', 'w',encoding='utf-8') as f:
            for row in res:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        t2 = time()
        print('Time Cost : %.1f seconds' % (t2 - t1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-build_vocab',action='store_true')
    parser.add_argument('-my_build_vocab',action='store_true')
    # parser.add_argument('-embed', type=str, default='data/100.w2v')
    parser.add_argument('-vocab', type=str, default='/content/drive/MyDrive/NLP/nlp_text_summarization/SummaRuNNer/mydata/embedding.npz')
    parser.add_argument('-word2id',type=str,default='/content/drive/MyDrive/NLP/nlp_text_summarization/SummaRuNNer/mydata/word2id.json')

    parser.add_argument('-worker_num',type=int,default=1)
    parser.add_argument('-source_dir', type=str, default='/content/drive/MyDrive/nlp_project/nlp_text_summarization/prep_data/')
    parser.add_argument('-output_dir', type=str, default='/content/drive/MyDrive/NLP/nlp_text_summarization/SummaRuNNer/mydata/')

    args = parser.parse_args()
    
    if args.build_vocab:
        build_vocab(args)
    elif args.my_build_vocab:
        my_build_vocab(args)
    else:
        my_build_dataset(args)
