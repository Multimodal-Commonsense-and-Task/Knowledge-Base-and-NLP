import string
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast, BertForMaskedLM
from colbert.parameters import DEVICE

from IsoScore.regularizers import istar
from IsoScore.training_utils import get_ci


def avg_cos(embs):
    embs = F.normalize(embs,dim=-1)
    scores = embs @ embs.permute(0,1,3,2) # (bsize, q_len, d_len, d_len)
    return torch.mean(scores)

def cond_isotropy(Q, D, cos, metric="avg_cos"):
    # Q: (bsize, q_len, 128)
    # D: (bsize, d_len, 128)
    # cos: (bsize, q_len, d_len)
    D = D.unsqueeze(dim=1) # (bsize, 1, d_len, 128)
    cos = cos.unsqueeze(dim=-1) # (bsize, q_len, d_len, 1)
    Q = Q.unsqueeze(dim=2) # (bsize, q_len, 1, 128)
    Qcos = cos * Q # (bsize, q_len, d_len, 128)
    embs = D - Qcos # (bsize, q_len, d_len, 128)
    return avg_cos(embs)


class ColBERT(BertPreTrainedModel):
    def __init__(self,
            config, 
            query_maxlen, 
            doc_maxlen, 
            mask_punctuation, 
            dim=128, 
            similarity_metric='cosine', 
            qidf=None, 
            regularizer=None, 
            reg_lambda=None,
            balanced=False,
            pmi=False,
        ):

        super(ColBERT, self).__init__(config)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}

        self.qidf = qidf
        self.regularizer = regularizer
        if self.regularizer in ["istar", "cosreg", "sensim"]:
            self.reg_lambda = reg_lambda
            if self.regularizer == "istar":
                self.reg = istar()
                self.zeta = 0.2
                self.C0 = torch.eye(dim).to(DEVICE)#.to(torch.float64)
                self.iso_loss = []
            elif self.regularizer == "cosreg":
                self.cosreg_loss = []

        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)
        self.lexical_linear = nn.Linear(config.hidden_size, dim, bias=False)

        self.init_weights()

    def forward(self, Q, D):
        # ColBERT-HIL
        if self.qidf:
            Q_ids = Q[0].to(DEVICE)
            D_ids = D[0].to(DEVICE)
            Q, lexical_Q = self.query(*Q)
            D, lexical_D = self.doc(*D)
            return self.score(Q, D, Q_ids, D_ids, lexical_Q, lexical_D)

        # ColBERT, IsoScore
        return self.score(self.query(*Q), self.doc(*D))

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)

        # ColBERT-HIL
        if self.qidf:
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            Q = outputs[0]
            Q = self.linear(Q)

            lexical_Q = outputs.embedding_output
            lexical_Q = self.lexical_linear(lexical_Q)
            lexical_Q = torch.nn.functional.normalize(lexical_Q, p=2, dim=2)

            return torch.nn.functional.normalize(Q, p=2, dim=2), lexical_Q
        
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        # ColBERT
        Q = outputs[0]
        Q = self.linear(Q)
        Q = torch.nn.functional.normalize(Q, p=2, dim=2)

        if self.regularizer == "cosreg":
            W = torch.matmul(Q.reshape(-1,128), Q.reshape(-1,128).T)
            N = W.shape[0]
            cosreg_loss = (torch.sum(W) - N) / (N**2)
            self.cosreg_loss.append(cosreg_loss)
        elif self.regularizer == "istar":
            points = torch.reshape(Q, (-1,128))
            batch_iso = self.reg.isoscore_star(points, self.C0, zeta=self.zeta, gpu_id=DEVICE)
            iso_score_loss = 1 - batch_iso
            self.iso_loss.append(iso_score_loss)

        return Q

    def doc(self, input_ids, attention_mask, keep_dims=True):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)

        # ColBERT-HIL
        if self.qidf:
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            D = outputs[0]
            D = self.linear(D)

            lexical_D = outputs.embedding_output
            lexical_D = self.lexical_linear(lexical_D)

            mask = torch.tensor(self.mask(input_ids), device=DEVICE).unsqueeze(2).float()
            D = D * mask
            D = torch.nn.functional.normalize(D, p=2, dim=2)
            lexical_D = lexical_D * mask
            lexical_D = torch.nn.functional.normalize(lexical_D, p=2, dim=2)

            if not keep_dims:
                D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
                D = [d[mask[idx]] for idx, d in enumerate(D)]
                lexical_D = lexical_D.cpu().to(dtype=torch.float16)
                lexical_D = [d[mask[idx]] for idx, d in enumerate(lexical_D)]
            return D, lexical_D
        
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        # ColBERT
        D = outputs[0]
        D = self.linear(D)
        mask = torch.tensor(self.mask(input_ids), device=DEVICE).unsqueeze(2).float()
        D = D * mask
        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if self.regularizer == "cosreg":
            W = torch.matmul(D.reshape(-1,128), D.reshape(-1,128).T)
            N = W.shape[0]
            cosreg_loss = (torch.sum(W) - N) / (N**2)
            self.cosreg_loss.append(cosreg_loss)
        elif self.regularizer == "istar":
            points = torch.reshape(D, (-1,128))
            batch_iso = self.reg.isoscore_star(points, self.C0, zeta=self.zeta, gpu_id=DEVICE)
            iso_score_loss = 1 - batch_iso
            self.iso_loss.append(iso_score_loss)


        if not keep_dims:
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]
        return D

    def score(self, Q, D, Q_ids=None, D_ids=None, lexical_Q=None, lexical_D=None):
        if self.similarity_metric == 'cosine':
            # ColBERT-HIL
            if self.qidf:
                assert Q_ids is not None and lexical_Q is not None and lexical_D is not None
                cb_dot = Q @ D.permute(0, 2, 1) # (bsize, q_len, d_len)
                lx_dot = lexical_Q @ lexical_D.permute(0, 2, 1) # (bsize, q_len, d_len)

                # cb_score = cb_dot.max(2).values
                # lx_score = lx_dot.max(2).values
                cb_score, cb_indices = cb_dot.max(2)
                lx_score, lx_indices = lx_dot.max(2)
                
                q_weights = []
                for ids in Q_ids:
                    q_weight = torch.tensor([self.qidf[tokid.item()] for tokid in ids])
                    q_weights.append(q_weight)
                q_weights = torch.stack(q_weights).to(lx_score.device)
                lx_score = q_weights * lx_score
            
                m_cb, std_cb = torch.mean(cb_score), torch.std(cb_score)
                m_lx, std_lx = torch.mean(lx_score), torch.std(lx_score)

                cb_score = (cb_score - m_cb) / std_cb
                lx_score = (lx_score - m_lx) / std_lx

                cb_score = cb_score.sum(1)
                lx_score = lx_score.sum(1)

                if self.regularizer == "sensim":
                    # sensim_loss = torch.mean(cb_dot) - torch.mean(lx_dot)
                    # sensim_loss = torch.mean(cb_dot) - torch.mean(lx_dot).detach()

                    sensim_loss = torch.mean(cb_dot) - torch.abs(torch.mean(lx_dot))
                    # sensim_loss = torch.mean(cb_dot)
                    # sensim_loss = -torch.abs(torch.mean(lx_dot))
                    
                    return (cb_score + lx_score) / 2, sensim_loss
                return (cb_score + lx_score) / 2
            
            # IsoScore
            if self.regularizer == "istar":
                assert len(self.iso_loss) == 2, len(self.iso_loss)
                iso_loss = torch.mean(torch.stack(self.iso_loss))
                self.iso_loss = []
                return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1), iso_loss
            # CosReg
            elif self.regularizer == "cosreg":
                assert len(self.cosreg_loss) == 2, len(self.cosreg_loss)
                cosreg_loss = torch.mean(torch.stack(self.cosreg_loss))
                self.cosreg_loss = []
                return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1), cosreg_loss
            # SenSim
            elif self.regularizer == "sensim":
                cb_dot = Q @ D.permute(0, 2, 1) # (bsize, q_len, d_len)
                # sensim_loss = torch.max(torch.mean(cb_dot), torch.tensor(0))
                # sensim_loss = torch.abs(torch.mean(cb_dot))
                half_bsize = Q.shape[0] // 2
                sensim_loss = cond_isotropy(Q[:half_bsize], D[:half_bsize], cos=cb_dot[:half_bsize], metric="avg_cos")

                return cb_dot.max(2).values.sum(1), sensim_loss
            # ColBERT
            else:
                return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask
