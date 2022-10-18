
import argparse


from model import *

from dataloader import get_data_loaders
from torchmetrics import F1Score



def evaluation(args, model, batches, ckpt_path=None):

    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))

    with torch.no_grad(): 
        model.eval()
        mode = 'test'
                
        WAF1 = F1Score(num_classes=args.n_classes, average="weighted")  # F1 score
    

        preds, labels = [], []
        for batch in batches:
            text_features, audio_features, visual_features, label, adj,s_mask, _, _  = batch

            text_features = text_features.to(args.device)
            audio_features = audio_features.to(args.device)
            visual_features = visual_features.to(args.device)
            label = label.to(args.device)
            adj = adj.to(args.device)
            s_mask = s_mask.to(args.device)

            log_prob = model(text_features, audio_features,visual_features,adj, s_mask, mode) # (B, N, C)

            label = label.cpu().numpy().tolist()
            pred = torch.argmax(log_prob, dim = 2).cpu().numpy().tolist() # (B,N) 
            preds += pred
            labels += label


        new_preds = []
        new_labels = []
        for i,label in enumerate(labels): 
            for j,l in enumerate(label):
                if l != -1: 
                    new_labels.append(l)
                    new_preds.append(preds[i][j])


        waf1 = WAF1(torch.tensor(new_preds),torch.tensor(new_labels))  # 默认 pred在前 truth在后


    return waf1.item()
  




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', default='IEMOCAP', type=str, help='dataset name, IEMOCAP or MELD')
    parser.add_argument('--hidden_dim', type=int, default=300) # meld 600  ie 300
    parser.add_argument('--gnn_layers', type=int, default=4, help='Number of gnn layers.') # meld 5  ie 4
    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')
    parser.add_argument('--modality', type=str, default='tav', help='ta,tav')
    parser.add_argument('--batch_size', type=int, default=1, metavar='BS', help='batch size') 

    parser.add_argument('--ckpt_index', type=int, default=1)

    args = parser.parse_args()



    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", args.device)

    if args.dataset_name == 'IEMOCAP':
        args.n_classes = 6
        args.text_emb_dim = 1024
        args.audio_emb_dim = 100
        args.video_emb_dim = 512

    elif args.dataset_name == 'MELD':
        args.n_classes = 7
        args.text_emb_dim = 1024
        args.audio_emb_dim = 300
        args.video_emb_dim = 342

    print('building model..')

    model = SAMGN(args)
    model.to(args.device)


    test_loader = get_data_loaders(
        dataset_name=args.dataset_name, split='test', batch_size=args.batch_size, num_workers=0, args=args)

    ckpt_path = './ckpts/'+args.dataset_name+'_'+args.modality+'_ckpt'+str(args.ckpt_index)+'.pt'

    waf1 = evaluation(args, model, test_loader, ckpt_path=ckpt_path)

    print("test set WA.F1: ", waf1)

    with open(f"{args.dataset_name}_{args.modality}_result.txt", 'a+') as f:
        f.write(str(waf1))
        f.write("\n")




    








