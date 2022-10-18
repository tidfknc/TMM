import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random

import pickle
def read_pickle(filename):
    try:
        with open(filename,'rb') as f:
            data = pickle.load(f)
    except:
        with open(filename,'rb') as f:
            data = pickle.load(f,encoding='latin1')
    return data

class MyDataset(Dataset):

    def __init__(self, dataset_name = 'IEMOCAP', split = 'train', args = None):
        self.args = args
        self.dataset_name = dataset_name

        if dataset_name == 'IEMOCAP':
            self.videoSpeakers, self.videoLabels, text_feature1024, audio_feature100, visual_feature512, \
             self.trainVid, self.testVid, self.validVid = read_pickle('./data/IEMOCAP.pkl')

     
            self.text_feature = text_feature1024
            self.audio_feature = audio_feature100
            self.visual_feature = visual_feature512

            self.iemocap_speaker_mapping = {'M': 0, 'F': 1}

        elif dataset_name == 'MELD':

            self.videoSpeakers, self.videoLabels, text_feature1024,audio_feature300, visual_feature342, \
                self.trainVid, self.testVid, self.validVid = read_pickle(
                './data/MELD.pkl')

            self.text_feature = text_feature1024
            self.audio_feature = audio_feature300
            self.visual_feature = visual_feature342


        self.data = self.read(split)

        self.len = len(self.data)

    def read(self, split):

        # process dialogue
        if split=='train':
            dialog_ids = self.trainVid
        elif split=='dev':
            dialog_ids = self.validVid
        elif split=='test':
            dialog_ids = self.testVid

        dialogs = []
        utterance_count = 0
        for dialog_id in dialog_ids:
            labels = self.videoLabels[dialog_id]
            utterance_count = utterance_count + len(labels)

            if self.dataset_name == 'IEMOCAP':
                speakers = [self.iemocap_speaker_mapping[speaker] for speaker in self.videoSpeakers[dialog_id]]
            elif self.dataset_name == 'MELD':
                speakers = [speaker.index(1) for speaker in self.videoSpeakers[dialog_id]]


            text_features = [item.tolist() for item in self.text_feature[dialog_id]]
            audio_features = [item.tolist() for item in self.audio_feature[dialog_id]]
            visual_features = [item.tolist() for item in self.visual_feature[dialog_id]]

            dialogs.append({
                'id':dialog_id,
                'labels': labels,
                'speakers': speakers,
                'text_features': text_features,
                'audio_features': audio_features,
                'visual_features': visual_features
            })


        random.shuffle(dialogs) # 打乱对话
        print(split+' dialogue num:')
        print(len(dialogs)) # 对话数量

        print(split+' utterance num:')
        print(utterance_count) # 话语数量
        return dialogs

    def __getitem__(self, index):  # 获取一个样本/ 对话
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
        '''
        return torch.FloatTensor(self.data[index]['text_features']), \
               torch.FloatTensor(self.data[index]['audio_features']), \
               torch.FloatTensor(self.data[index]['visual_features']), \
               torch.LongTensor(self.data[index]['labels']),\
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['id']

    def __len__(self):
        return self.len

    def get_adj(self, lengths, max_dialog_len):
        '''
        get adj matrix
        :param speakers:  (B, N)
        :param max_dialog_len:
        :return:
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
        '''
        adj = []
        for length in lengths: 
            a = torch.zeros(max_dialog_len, max_dialog_len) 
            a[:length,:length] = 1
            adj.append(a)
        return torch.stack(adj)

    def get_s_mask(self, speakers, max_dialog_len):
        '''
        :param speakers:
        :param max_dialog_len:
        :return:
         s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
      
        '''
        s_mask = []

        for speaker in speakers: 
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype = torch.long) # （N,N）
            for i in range(len(speaker)): 
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i,j] = 1
            s_mask.append(s)
        return torch.stack(s_mask)

    def collate_fn(self, data):  
        '''
        :param data:
            text_features, audio_features, visual_features, labels, speakers, length,id
        :return:
            text_features: (B, N, D) padded
            audio_features: (B, N, D) padded
            visual_features: (B, N, D) padded
            labels: (B, N) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            lengths: (B, )
        '''
        max_dialog_len = max([d[5] for d in data]) # batch 中 对话的最大长度 N
        text_features = pad_sequence([d[0] for d in data], batch_first = True) # (B, N, D)
        audio_features = pad_sequence([d[1] for d in data], batch_first=True)  # (B, N, D)
        visual_features = pad_sequence([d[2] for d in data], batch_first=True)  # (B, N, D)

        labels = pad_sequence([d[3] for d in data], batch_first = True, padding_value = -1) # (B, N ) label 填充值为 -1

        adj = self.get_adj([d[5] for d in data], max_dialog_len) 

        s_mask = self.get_s_mask([d[4] for d in data], max_dialog_len) # （B, N, N）

        lengths = torch.LongTensor([d[5] for d in data]) 
        speakers = pad_sequence([torch.LongTensor(d[4]) for d in data], batch_first = True, padding_value = -1) # (B, N) speaker 填充值为 -1

        return text_features, audio_features,visual_features,labels, adj,s_mask,lengths, speakers
