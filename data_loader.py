import sys
import torch 
import torch.utils.data as data
import torch.nn as nn
import tables
import json
import random
import numpy as np
import pickle
import nltk
from nltk.corpus import wordnet 
from utils import PAD_ID, SOS_ID, EOS_ID, UNK_ID, indexes2sent, DEL_ID

    
class CodeSearchDataset(data.Dataset):
    """
    Dataset that has only positive samples.
    """
    def __init__(self, data_dir, f_name, max_name_len, f_api, max_api_len, 
                 f_tokens, max_tok_len, f_descs=None, max_desc_len=None):
        self.max_name_len=max_name_len
        self.max_api_len=max_api_len
        self.max_tok_len=max_tok_len
        self.max_desc_len=max_desc_len

        num = 1000000
        # 1. Initialize file path or list of file names.
        """read training data(list of int arrays) from a hdf5 file"""
        self.training=False
        print("loading data...")
        table_name = tables.open_file(data_dir+f_name)
        self.names = table_name.get_node('/phrases')[:].astype(np.long)
        self.idx_names = table_name.get_node('/indices')[:num]
        table_api = tables.open_file(data_dir+f_api)
        self.apis = table_api.get_node('/phrases')[:].astype(np.long)
        self.idx_apis = table_api.get_node('/indices')[:num]
        table_tokens = tables.open_file(data_dir+f_tokens)
        self.tokens = table_tokens.get_node('/phrases')[:].astype(np.long)
        self.idx_tokens = table_tokens.get_node('/indices')[:num]
        if f_descs is not None:
            self.training=True
            table_desc = tables.open_file(data_dir+f_descs)
            self.descs = table_desc.get_node('/phrases')[:].astype(np.long)
            self.idx_descs = table_desc.get_node('/indices')[:num]
        
        assert self.idx_names.shape[0] == self.idx_apis.shape[0]
        assert self.idx_apis.shape[0] == self.idx_tokens.shape[0]
        if f_descs is not None:
            assert self.idx_names.shape[0]==self.idx_descs.shape[0]
        self.data_len = self.idx_names.shape[0]

        with open('data/github/vocab.desc.json') as f:
            self.vocab = json.load(f)
        
        self.id2words = {}
        for i, k in enumerate(self.vocab):
            self.id2words[i] = k 

        self.vocab['[DEL]'] = 10000
        self.id2words[10000] = '[DEL]'

        self.max_vocab_size = 20000

        # self.tok_aug_dels = []
        # self.tok_aug_del_lens = []
        # self.tok_aug_swaps = []
        # self.tok_aug_swap_lens = []
        #self.toks = []
        #self.tok_lens = []

        # code token augmentation
        # for offset in range(len(self.idx_tokens)):
        #     length, pos = self.idx_tokens[offset]['length'], self.idx_tokens[offset]['pos']
        #     tok_len = min(int(length), self.max_tok_len)
        #     tok = self.tokens[pos:pos+tok_len]
            
        #     tok_aug_del = self.random_deletion(tok, 0.4)
        #     tok_aug_del_len = min(int(len(tok_aug_del)), self.max_tok_len)

        #     tok_aug_swap = self.random_swap(tok, 1)
        #     tok_aug_swap_len = min(int(len(tok_aug_swap)), self.max_tok_len)
            
        #     tok_aug_del = self.pad_seq(tok_aug_del, self.max_tok_len)
        #     tok_aug_swap = self.pad_seq(tok_aug_swap, self.max_tok_len)
            
        #     tok = self.pad_seq(tok, self.max_tok_len)
            
        #     self.tok_aug_dels.append(tok_aug_del)
        #     self.tok_aug_del_lens.append(tok_aug_del_len)
        #     self.tok_aug_swaps.append(tok_aug_swap)
        #     self.tok_aug_swap_lens.append(tok_aug_swap_len)
        #     # self.toks.append(tok)
        #     # self.tok_lens.append(tok_len)
        
        # print("Code token augmentation finished!")

        # Description augmentation
        # self.nl_aug_dels = []
        # self.nl_aug_del_lens = []
        # self.nl_aug_subs = []
        # self.nl_aug_sub_lens = []
        # self.good_descs = []
        # self.good_desc_lens = []

        # nltk.download('wordnet')
        # for offset in range(len(self.idx_descs)):
        #     length, pos = self.idx_descs[offset]['length'], self.idx_descs[offset]['pos']
        #     good_desc_len = min(int(length), self.max_desc_len)
        #     good_desc = self.descs[pos:pos+good_desc_len]
            
        #     nl_aug_del = self.random_deletion(good_desc, 0.4)
        #     nl_aug_del_len = min(int(len(nl_aug_del)), self.max_desc_len)

        #     nl_aug_sub = self.synonym_replacement(good_desc, 0.7)
        #     nl_aug_sub_len = min(int(len(nl_aug_sub)), self.max_desc_len)
            
        #     nl_aug_del = self.pad_seq(nl_aug_del, self.max_desc_len)
        #     nl_aug_sub = self.pad_seq(nl_aug_sub, self.max_desc_len)
            
        #     good_desc = self.pad_seq(good_desc, self.max_desc_len)
            
        #     self.nl_aug_dels.append(nl_aug_del)
        #     self.nl_aug_del_lens.append(nl_aug_del_len)
        #     self.nl_aug_subs.append(nl_aug_sub)
        #     self.nl_aug_sub_lens.append(nl_aug_sub_len)
        #     self.good_descs.append(good_desc)
        #     self.good_desc_lens.append(good_desc_len)
        
        #     if(good_desc_len== 0 or nl_aug_del_len == 0 or nl_aug_sub_len == 0):
        #       print(good_desc)
        #       print(nl_aug_del)
        #       print(nl_aug_sub)
        #print("Description augmentation finished!")
        print("{} entries".format(self.data_len))
        
    def pad_seq(self, seq, maxlen):
        if len(seq)<maxlen:
            # !!!!! numpy appending is slow. Try to optimize the padding
            seq=np.append(seq, [PAD_ID]*(maxlen-len(seq)))
        seq=seq[:maxlen]
        return torch.LongTensor(seq)
    
    def __getitem__(self, offset):          
        length, pos = self.idx_names[offset]['length'], self.idx_names[offset]['pos']
        name_len=min(int(length),self.max_name_len) 
        name = self.names[pos: pos+name_len]
        name = self.pad_seq(name, self.max_name_len)
        
        length, pos = self.idx_apis[offset]['length'], self.idx_apis[offset]['pos']
        api_len = min(int(length), self.max_api_len)
        apiseq = self.apis[pos:pos+api_len]
        apiseq = self.pad_seq(apiseq, self.max_api_len)
        
        length, pos = self.idx_tokens[offset]['length'], self.idx_tokens[offset]['pos']
        tok_len = min(int(length), self.max_tok_len)
        tok = self.tokens[pos:pos+tok_len]
            
        tok_aug_del = self.random_deletion(tok, 0.4)
        tok_aug_del_len = min(int(len(tok_aug_del)), self.max_tok_len)

        tok_aug_swap = self.random_swap(tok, 1)
        tok_aug_swap_len = min(int(len(tok_aug_swap)), self.max_tok_len)
            
        tok_aug_del = self.pad_seq(tok_aug_del, self.max_tok_len)
        tok_aug_swap = self.pad_seq(tok_aug_swap, self.max_tok_len)
            
        tokens = self.pad_seq(tok, self.max_tok_len)
            
        #tok_len = self.tok_lens[offset]
        #tokens = self.toks[offset]
        # tok_aug_del = self.tok_aug_dels[offset]
        # tok_aug_del_len = self.tok_aug_del_lens[offset]
        # tok_aug_swap = self.tok_aug_swaps[offset]
        # tok_aug_swap_len = self.tok_aug_swap_lens[offset]

        if self.training:
            length, pos = self.idx_descs[offset]['length'], self.idx_descs[offset]['pos']
            good_desc_len = min(int(length), self.max_desc_len)
            good_desc = self.descs[pos:pos+good_desc_len]
            
            nl_aug_del = self.random_deletion(good_desc, 0.4)
            nl_aug_del_len = min(int(len(nl_aug_del)), self.max_desc_len)

            nl_aug_sub = self.synonym_replacement(good_desc, 0.7)
            nl_aug_sub_len = min(int(len(nl_aug_sub)), self.max_desc_len)
            
            nl_aug_del = self.pad_seq(nl_aug_del, self.max_desc_len)
            nl_aug_sub = self.pad_seq(nl_aug_sub, self.max_desc_len)
            
            good_desc = self.pad_seq(good_desc, self.max_desc_len)
            
            # good_desc_len = self.good_desc_lens[offset]
            # good_desc = self.good_descs[offset]
            # nl_aug_del = self.nl_aug_dels[offset]
            # nl_aug_del_len = self.nl_aug_del_lens[offset]
            # nl_aug_sub = self.nl_aug_subs[offset]
            # nl_aug_sub_len = self.nl_aug_sub_lens[offset]
    
            rand_offset=random.randint(0, self.data_len-1)
            length, pos = self.idx_descs[rand_offset]['length'], self.idx_descs[rand_offset]['pos']
            bad_desc_len=min(int(length), self.max_desc_len)
            bad_desc = self.descs[pos:pos+bad_desc_len]
            bad_desc = self.pad_seq(bad_desc, self.max_desc_len)
            
            if(bad_desc_len == 0):
              print(bad_desc)
            return name, name_len, apiseq, api_len, tokens, tok_len, good_desc, good_desc_len, bad_desc, bad_desc_len, nl_aug_del, nl_aug_del_len, nl_aug_sub, nl_aug_sub_len, tok_aug_del, tok_aug_del_len, tok_aug_swap, tok_aug_swap_len
        return name, name_len, apiseq, api_len, tokens, tok_len
        
    def __len__(self):
        return self.data_len
    
    def random_deletion(self, words, p):
        if len(words) == 1:
            return words

        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)
            else:
                new_words.append(DEL_ID)

        return new_words

    def get_synonyms(self, word):
        """
        Get synonyms of a word
        """
        synonyms = set()
        
        for syn in wordnet.synsets(word): 
            for l in syn.lemmas(): 
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym) 
        
        if word in synonyms:
            synonyms.remove(word)
        
        return list(synonyms)

    def synonym_replacement(self, words, p):
        words = [self.id2words[i] for i in words]
        num_replaced = 0
        
        idx = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                synonyms = self.get_synonyms(word)
                if len(synonyms) >= 1:
                    synonym = random.choice(list(synonyms))
                    if synonym in self.vocab:
                        idx.append(self.vocab[synonym])
                    elif len(self.vocab) < self.max_vocab_size:
                        syn_idx = len(self.vocab)
                        idx.append(syn_idx)
                        self.vocab[synonym] = syn_idx
                        self.id2words[syn_idx] = synonym
                    else:
                        check = 0
                        for syn in synonyms:
                            if(syn in self.vocab):
                                idx.append(self.vocab[syn])
                                check = 1
                                break
                        if check == 0:
                            idx.append(self.vocab[word])
                else:
                  idx.append(self.vocab[word])
            else:
                idx.append(self.vocab[word])

        return idx

    def swap_word(self, new_words):   
        random_idx_1 = random.randint(0, len(new_words)-1)
        random_idx_2 = random_idx_1
        counter = 0
        
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words)-1)
            counter += 1
            
            if counter > 3:
                return new_words
        
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
        return new_words

    def random_swap(self,words, n):
        #words = words.split()
        new_words = words.copy()
        
        for _ in range(n):
            new_words = self.swap_word(new_words)
        return new_words

def load_dict(filename):
    return json.loads(open(filename, "r").readline())
    #return pickle.load(open(filename, 'rb')) 

def load_vecs(fin):         
    """read vectors (2D numpy array) from a hdf5 file"""
    h5f = tables.open_file(fin)
    h5vecs= h5f.root.vecs
    
    vecs=np.zeros(shape=h5vecs.shape,dtype=h5vecs.dtype)
    vecs[:]=h5vecs[:]
    h5f.close()
    return vecs
        
def save_vecs(vecs, fout):
    fvec = tables.open_file(fout, 'w')
    atom = tables.Atom.from_dtype(vecs.dtype)
    filters = tables.Filters(complib='blosc', complevel=5)
    ds = fvec.create_carray(fvec.root,'vecs', atom, vecs.shape,filters=filters)
    ds[:] = vecs
    print('done')
    fvec.close()

if __name__ == '__main__':
    input_dir='./data/github/'
    train_set=CodeSearchDataset(input_dir, 'train.name.h5', 6, 'train.apiseq.h5', 20, 'train.tokens.h5', 30, 'train.desc.h5', 30)
    train_data_loader=torch.utils.data.DataLoader(dataset=train_set, batch_size=1, shuffle=False, num_workers=1)
    use_set=CodeSearchDataset(input_dir, 'use.name.h5', 6, 'use.apiseq.h5', 20, 'use.tokens.h5', 30)
    use_data_loader=torch.utils.data.DataLoader(dataset=use_set, batch_size=1, shuffle=False, num_workers=1)
    vocab_api = load_dict(input_dir+'vocab.apiseq.json')
    vocab_name = load_dict(input_dir+'vocab.name.json')
    vocab_tokens = load_dict(input_dir+'vocab.tokens.json')
    vocab_desc = load_dict(input_dir+'vocab.desc.json')
    
    print('============ Train Data ================')
    k=0
    for batch in train_data_loader:
        batch = tuple([t.numpy() for t in batch])
        name, name_len, apiseq, api_len, tokens, tok_len, good_desc, good_desc_len, bad_desc, bad_desc_len = batch
        k+=1
        if k>20: break
        print('-------------------------------')
        print(indexes2sent(name, vocab_name))
        print(indexes2sent(apiseq, vocab_api))
        print(indexes2sent(tokens, vocab_tokens))
        print(indexes2sent(good_desc, vocab_desc))
        
    print('\n\n============ Use Data ================')
    k=0
    for batch in use_data_loader:
        batch = tuple([t.numpy() for t in batch])
        name, name_len, apiseq, api_len, tokens, tok_len = batch
        k+=1
        if k>20: break
        print('-------------------------------')
        print(indexes2sent(name, vocab_name))
        print(indexes2sent(apiseq, vocab_api))
        print(indexes2sent(tokens, vocab_tokens))