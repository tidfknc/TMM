import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_logic(alpha, adj):
    '''
    performing mask logic with adj
    :param alpha:
    :param adj:
    :return:
    '''
    return alpha - (1 - adj) * 1e30


class GatingLayer(nn.Module):

    def __init__(self,args):
        super().__init__()
        self.args = args

        self.fc1 = nn.Linear(args.hidden_dim,args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim,args.hidden_dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.fc3 = nn.Linear(args.hidden_dim,args.hidden_dim)
        self.fc4 = nn.Linear(args.hidden_dim,args.hidden_dim,bias=False)

    def forward(self,h_intra,h_inter):
        '''

        :param h_intra:  (B,3,N,hidden_dim)
        :param h_inter: (B,3,N,hidden_dim)
        :return:
        '''
        p = self.tanh(self.fc1(h_intra)) #  (B,3,N,hidden_dim)
        q = self.tanh(self.fc2(h_inter)) #  (B,3,N,hidden_dim)

        s = self.sigmoid(self.fc3(h_intra)+self.fc4(h_inter)) # (B,3,N,hidden_dim)

        h_merge = s*p + (1-s)*q

        return h_merge



class GCN(nn.Module):
    '''
    不考虑说话人关系
    '''

    def __init__(self, args):
        super().__init__()

        self.linear = nn.Linear(args.hidden_dim * 2, 1)  # 计算attention权重

    def forward(self, Q, K, V, adj):
        '''
        imformation gatherer with linear attention
        :param Q: (B, N, D) # query utterances
        :param K: (B, N, D) # context
        :param V: (B, N, D) # context
        :param adj: (B, N, N) # the adj matrix of entire conversation
        :return:
        '''
        B = K.size()[0]
        N = K.size()[1]  # max utterance num
        Q1 = Q.repeat_interleave(N,1) # (B,N*N,D)
        K1 = K.repeat(1,N,1)  # (B,N*N,D)

        X = torch.cat((Q1, K1), dim=2)  # (B, N*N, 2D)

        alpha = self.linear(X).squeeze(-1) # (B,N*N,1)
        alpha = alpha.view(B,N,N) # (B,N,N)

 
        alpha = mask_logic(alpha, adj)  # (B, N, N)
     

        attn_weight = F.softmax(alpha, dim=2)  # (B, N, N)
      

        attn_sum = torch.bmm(attn_weight, V)  # (B, N, D)

        return attn_sum


class SRGCN(nn.Module):
    '''
    use linear to avoid OOM
    H_i = alpha_ij(W_rH_j)
    alpha_ij = attention(H_i, H_j)
    考虑说话人关系
    '''

    def __init__(self, args):
        super().__init__()

        self.args = args

        self.linear = nn.Linear(args.hidden_dim * 2, 1)  # 计算attention权重
        self.Wr0 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)  # 相同说话人对应的权重
        self.Wr1 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)  # 不同说话人对应的权重

        self.Wq = nn.Linear(args.hidden_dim,args.hidden_dim)
        self.Wk = nn.Linear(args.hidden_dim,args.hidden_dim)


    def structure_learning(self,Q,K,mode):
        '''
        :param Q: (B, N, D) # query utterances
        :param K: (B, N, D) # context
        :param adj: (B, N, N) # the adj matrix of entire conversation
        :return:
        '''
        Q1 = self.Wq(Q)
        K1 = self.Wk(K)
        logits = torch.matmul(Q1,K1.permute(0,2,1)) # (B,N,N)

        if mode == 'train':
            sample_prob = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(self.args.tau, logits=logits)  # (0,1)
            y = sample_prob.sample()  # (0,1)
            y_hard = (y > 0.5).to(y.dtype) # 0.0 or 1.0
            y = (y_hard - y).detach() + y
        elif mode == 'test':
            prob = torch.sigmoid(logits) # (B,N,N)  (0,1)
            y = (prob>0.5).to(prob.dtype)

        return y

    def forward(self, Q, K, V, adj, s_mask,mode):
        '''
        imformation gatherer with linear attention
        :param Q: (B, N, D) # query utterances
        :param K: (B, N, D) # context
        :param V: (B, N, D) # context
        :param adj: (B, N, N) # the adj matrix of entire conversation
        :param s_mask: (B, N, N)
        :return:
        '''
        sample_action = self.structure_learning(Q, K, mode)  # (B,N,N) 参与运算 以便反传更新

        B = K.size()[0]
        N = K.size()[1]  # max utterance num
        Q1 = Q.repeat_interleave(N,1) # (B,N*N,D)
        K1 = K.repeat(1,N,1)  # (B,N*N,D)

        X = torch.cat((Q1, K1), dim=2)  # (B, N*N, 2D)

        alpha = self.linear(X).squeeze(-1) # (B,N*N,1)
        alpha = alpha.view(B,N,N) # (B,N,N)  logits

        logits_exp = torch.exp(alpha)  # (B,N,N)
        logits_exp = logits_exp*adj  # (B,N,N) 忽略掉填充部分
        logits_exp = logits_exp*sample_action  # (B,N,N) 忽略掉 结构学习 discard的边

        logits_sum = torch.sum(logits_exp, dim=-1, keepdim=True)  # (B,N,1)

        attn_weight = logits_exp / (logits_sum + 1e-6)  # (B,N,N)


        V0 = self.Wr0(V)  # (B, N, D)
        V1 = self.Wr1(V)  # (B, N, D)

        attn_weight1 = s_mask*attn_weight
        attn_weight2 = (1-s_mask)*attn_weight
        attn_sum1 = torch.bmm(attn_weight1, V0)  # (B, N, D)
        attn_sum2 = torch.bmm(attn_weight2, V1)  # (B, N, D)
        attn_sum = attn_sum1 + attn_sum2


        return attn_sum






