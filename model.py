

from model_utils import *


class SAMGN(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        # gcn layer
        self.dropout = nn.Dropout(0.0)
        self.relu = nn.ReLU()

        # 各个模态特征转换 embedding -> hidden
        self.text_modal_encoder = nn.Linear(args.text_emb_dim, args.hidden_dim)
        self.audio_modal_encoder = nn.Linear(args.audio_emb_dim, args.hidden_dim)
        self.video_modal_encoder = nn.Linear(args.video_emb_dim, args.hidden_dim)

        self.gnn_layers = args.gnn_layers

        text_gcns = []
        for _ in range(args.gnn_layers):
            text_gcns += [SRGCN(args)] 
        self.text_gcns = nn.ModuleList(text_gcns)

        audio_gcns = []
        for _ in range(args.gnn_layers):
            audio_gcns += [SRGCN(args)]
        self.audio_gcns = nn.ModuleList(audio_gcns)

        video_gcns = []
        for _ in range(args.gnn_layers):
            video_gcns += [SRGCN(args)]
        self.video_gcns = nn.ModuleList(video_gcns)


        inter_gcns = []
        for _ in range(args.gnn_layers):
            inter_gcns += [GCN(args)]  # inter 
        self.inter_gcns = nn.ModuleList(inter_gcns)

        gatings = []
        for _ in range(args.gnn_layers):
            gatings += [GatingLayer(args)]
        self.gatings = nn.ModuleList(gatings)


        text_in_dim = args.hidden_dim * (args.gnn_layers + 1) + args.text_emb_dim
        audio_in_dim = args.hidden_dim * (args.gnn_layers + 1) + args.audio_emb_dim
        video_in_dim = args.hidden_dim * (args.gnn_layers + 1) + args.video_emb_dim


        self.text_out_trans = nn.Linear(text_in_dim, args.hidden_dim)
        self.audio_out_trans = nn.Linear(audio_in_dim, args.hidden_dim)
        self.video_out_trans = nn.Linear(video_in_dim, args.hidden_dim)


        output_layers = []
        for _ in range(args.mlp_layers - 1):
            output_layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        output_layers += [self.dropout]
        output_layers += [nn.Linear(args.hidden_dim, args.n_classes)]

        self.out_mlp = nn.Sequential(*output_layers)



    def forward(self, text_features,audio_features,visual_features, adj, s_mask, mode):
        '''
        :param xx_features: (B, N, D)
        :param adj: (B, N, N)
        :param s_mask: (B, N, N)
        :return:
        '''

        Num = text_features.size(1)
        batch_size = text_features.size(0)

        if self.args.modality == 'ta':
     
            inter_adj = (torch.ones(batch_size * Num, len(self.args.modality), len(self.args.modality)) - torch.eye(
                len(self.args.modality))).to(self.args.device)

            H1 = self.relu(self.text_modal_encoder(text_features))  # (B,N,hidden)
            H2 = self.relu(self.audio_modal_encoder(audio_features))
            A = [H1]
            B = [H2]

            H4 = torch.cat([H1.unsqueeze(1), H2.unsqueeze(1)], dim=1).permute(0, 2, 1, 3).reshape(
                batch_size * Num, len(self.args.modality), self.args.hidden_dim)  # (B*N,2,hidden_dim)
            D = [H4]

            for l in range(self.args.gnn_layers):

                M1 = self.text_gcns[l](A[l], A[l], A[l], adj, s_mask,mode)
                M2 = self.audio_gcns[l](B[l], B[l], B[l], adj, s_mask,mode)

                M4 = self.inter_gcns[l](D[l], D[l], D[l],
                                        inter_adj,
                                        )  # (B*N,2,hidden_dim)

                M5 = M4.reshape(batch_size, Num, len(self.args.modality), self.args.hidden_dim).permute(0, 2, 1,
                                                                                                        3)  # (B,2,N,hidden_dim)

                M6 = torch.cat([M1.unsqueeze(1), M2.unsqueeze(1)], dim=1)  # (B,2,N,hidden_dim)


                M7 = self.gatings[l](M6, M5)


                N1, N2  = torch.split(M7, 1, dim=1)  # (B,N,D)

                A.append(N1.squeeze(1))
                B.append(N2.squeeze(1))

                D.append(
                    M7.permute(0, 2, 1, 3).reshape(batch_size * Num, len(self.args.modality), self.args.hidden_dim))


            
            A.append(text_features)
            B.append(audio_features)


            A = torch.cat(A, dim=2)
            B = torch.cat(B, dim=2)


            A = self.relu(self.text_out_trans(A))  # (B,N,hidden_dim)
            B = self.relu(self.audio_out_trans(B))


    
            C = (A + B)/2.0   # (B,N,hidden_dim) mean pooling

            logits = self.out_mlp(C)
            return logits

       
        elif self.args.modality == 'tav':
        
            inter_adj = (torch.ones(batch_size*Num,len(self.args.modality),len(self.args.modality))-torch.eye(len(self.args.modality))).to(self.args.device)

            H1 = self.relu(self.text_modal_encoder(text_features))  # (B,N,hidden)
            H2 = self.relu(self.audio_modal_encoder(audio_features))
            H3 = self.relu(self.video_modal_encoder(visual_features))


            A = [H1]
            B = [H2]
            C = [H3]


            H4 = torch.cat([H1.unsqueeze(1),H2.unsqueeze(1),H3.unsqueeze(1)],dim=1).permute(0,2,1,3).reshape(batch_size*Num,len(self.args.modality),self.args.hidden_dim) # (B*N,3,hidden_dim)
            D = [H4]

            for l in range(self.args.gnn_layers):

                M1 = self.text_gcns[l](A[l], A[l], A[l], adj, s_mask,mode)
                M2 = self.audio_gcns[l](B[l], B[l], B[l], adj, s_mask,mode)
                M3 = self.video_gcns[l](C[l], C[l], C[l], adj, s_mask,mode)
                M4 = self.inter_gcns[l](D[l],D[l],D[l],
                                        inter_adj,
                                        ) # (B*N,3,hidden_dim)

                M5 = M4.reshape(batch_size,Num,len(self.args.modality),self.args.hidden_dim).permute(0,2,1,3) # (B,3,N,hidden_dim)

                M6 = torch.cat([M1.unsqueeze(1),M2.unsqueeze(1),M3.unsqueeze(1)],dim=1)  # (B,3,N,hidden_dim)

    
                M7 = self.gatings[l](M6,M5)
              

                N1,N2,N3 = torch.split(M7,1,dim=1) # (B,N,D)

                A.append(N1.squeeze(1))
                B.append(N2.squeeze(1))
                C.append(N3.squeeze(1))

                D.append(M7.permute(0,2,1,3).reshape(batch_size*Num,len(self.args.modality),self.args.hidden_dim))


            A.append(text_features)
            B.append(audio_features)
            C.append(visual_features)

            A = torch.cat(A, dim=2)
            B = torch.cat(B, dim=2)
            C = torch.cat(C, dim=2)

        
            A = self.relu(self.text_out_trans(A))  # (B,N,hidden_dim)
            B = self.relu(self.audio_out_trans(B))
            C = self.relu(self.video_out_trans(C))

            
         
            E = (A + B +C)/3.0   # (B,N,hidden_dim) mean pooling

            logits = self.out_mlp(E)
            return logits
        

