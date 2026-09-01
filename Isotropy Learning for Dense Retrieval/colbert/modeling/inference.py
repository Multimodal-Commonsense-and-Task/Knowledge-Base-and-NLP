import torch

from colbert.modeling.colbert import ColBERT
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from colbert.utils.amp import MixedPrecisionManager
from colbert.parameters import DEVICE

DEBUG=False

class ModelInference():
    def __init__(self, colbert: ColBERT, amp=False):
        assert colbert.training is False

        self.colbert = colbert
        self.skiplist = torch.Tensor([id for id in self.colbert.skiplist.keys() if isinstance(id, int)])
        self.query_tokenizer = QueryTokenizer(colbert.query_maxlen)
        self.doc_tokenizer = DocTokenizer(colbert.doc_maxlen)

        self.amp_manager = MixedPrecisionManager(amp)

    def query(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                Q = self.colbert.query(*args, **kw_args)

                # IsoBERT
                if self.colbert.qidf:
                    Q, lexical_Q = Q
                    return Q.cpu() if to_cpu else Q, lexical_Q.cpu() if to_cpu else lexical_Q
                # ColBERT
                else:
                    return Q.cpu() if to_cpu else Q

    def doc(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                D = self.colbert.doc(*args, **kw_args)

                # IsoBERT
                if self.colbert.qidf:
                    D, lexical_D = D
                    return D.cpu() if to_cpu else D, lexical_D.cpu() if to_cpu else lexical_D
                # ColBERT
                else:
                    return D.cpu() if to_cpu else D

    def queryFromText(self, queries, bsize=None, to_cpu=False, with_ids=False):
        if bsize:
            batches = self.query_tokenizer.tensorize(queries, bsize=bsize)
            batchesEmbs = [self.query(input_ids, attention_mask, to_cpu=to_cpu) for input_ids, attention_mask in batches]

            # IsoBERT
            if self.colbert.qidf:
                lexical_batchesEmbs = [batch[1] for batch in batchesEmbs]
                batchesEmbs = [batch[0] for batch in batchesEmbs]
                if with_ids:
                    return torch.cat(batchesEmbs), torch.cat(lexical_batchesEmbs), torch.cat([ids for ids, _ in batches]), torch.cat([masks for _, masks in batches])
                return torch.cat(batchesEmbs), torch.cat(lexical_batchesEmbs)
            # ColBERT
            else:
                if with_ids:
                    return torch.cat(batchesEmbs), torch.cat([ids for ids, _ in batches]), torch.cat([masks for _, masks in batches])
                return torch.cat(batchesEmbs)

        input_ids, attention_mask = self.query_tokenizer.tensorize(queries)
        query_embs = self.query(input_ids, attention_mask)
        lexical_query_embs = query_embs[1]
        query_embs = query_embs[0]

        if with_ids:
            return (query_embs, lexical_query_embs, input_ids, attention_mask)
        return query_embs, lexical_query_embs

    def docFromText(self, docs, bsize=None, keep_dims=True, to_cpu=False, with_ids=False):
        if bsize:
            #print("docFromText on %d documents" % len(docs))
            batch_ids, reverse_indices = self.doc_tokenizer.tensorize(docs, bsize=bsize)
            # batch_ids contain batches; each batch is a 2-tuple, of which the left is
            # the ids of each document, and the right is the masks of each document
            #print("tokens doc 0: %d" % len(batch_ids[0][0][0]))
            #print("total tokens %d" % sum([len(d) for ids, mark in batch_ids for d in ids]))
            #batch_ids = [ input_ids for input_ids in batches]

            #print("batch_ids len=%d" % len(batch_ids))
            #print("reverse_indices.shape=" + str(reverse_indices.shape))
            
            batches = [self.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)
                       for input_ids, attention_mask in batch_ids]
            #print("batches len = %d " % len(batches))


            # IsoBERT
            if self.colbert.qidf:
                lexical_batches = [batch[1] for batch in batches]
                batches = [batch[0] for batch in batches]

                if keep_dims:
                    D = _stack_3D_tensors(batches)
                    lexical_D = _stack_3D_tensors(lexical_batches)

                    if with_ids:
                        Dids = _stack_3D_tensors(batch_ids)
                        return D[reverse_indices], lexical_D[reverse_indices], Dids
                    return D[reverse_indices], lexical_D[reverse_indices]
                
                D = [d for batch in batches for d in batch]
                lexical_D = [d for batch in lexical_batches for d in batch]

                if with_ids:
                    D_i = [ d[(mask > 0) & (d != 0)] for input_ids, attention_masks in batch_ids for d, mask in zip(input_ids,attention_masks) ]
                    # remove skiplist tokens from ids
                    if len(self.colbert.skiplist) > 0:
                        D_i = [d[[token not in self.colbert.skiplist for token in d.tolist()]] for d in D_i]

                    left = [D[idx] for idx in reverse_indices.tolist()]
                    right = [D_i[idx] for idx in reverse_indices.tolist()]
                    lexical_left = [lexical_D[idx] for idx in reverse_indices.tolist()]
                    return left, lexical_left, right
                return [D[idx] for idx in reverse_indices.tolist()], [lexical_D[idx] for idx in reverse_indices.tolist()]

            # ColBERT
            else:
                if keep_dims:
                    D = _stack_3D_tensors(batches)
                    if with_ids:
                        Dids = _stack_3D_tensors(batch_ids)
                        return D[reverse_indices], Dids
                    return D[reverse_indices]
                
                D = [d for batch in batches for d in batch]
                if with_ids:
                    D_i = [ d[(mask > 0) & (d != 0)] for input_ids, attention_masks in batch_ids for d, mask in zip(input_ids,attention_masks) ]
                    if len(self.colbert.skiplist) > 0:
                        D_i = [d[[token not in self.colbert.skiplist for token in d.tolist()]] for d in D_i]
                    left = [D[idx] for idx in reverse_indices.tolist()]
                    right = [D_i[idx] for idx in reverse_indices.tolist()]
                    return left, right
                return [D[idx] for idx in reverse_indices.tolist()]

            """
            if keep_dims:
                D = _stack_3D_tensors(batches)
                lexical_D = _stack_3D_tensors(lexical_batches)

                if with_ids:
                    Dids = _stack_3D_tensors(batch_ids)
                    return D[reverse_indices], lexical_D[reverse_indices], Dids
                return D[reverse_indices], lexical_D[reverse_indices]

            #print(batches[0][0])
            D = [d for batch in batches for d in batch]
            lexical_D = [d for batch in lexical_batches for d in batch]
            #print("lenD = %d " % len(D))
            if with_ids:
                D_i = [ d[(mask > 0) & (d != 0)] for input_ids, attention_masks in batch_ids for d, mask in zip(input_ids,attention_masks) ]
                # remove skiplist tokens from ids
                if len(self.colbert.skiplist) > 0:
                    D_i = [d[[token not in self.colbert.skiplist for token in d.tolist()]] for d in D_i]

                if DEBUG:
                    docid=-1
                    #for each batch
                    for embs, (input_ids, attention_masks) in zip(batches, batch_ids):
                        #for each document 
                        for emb, ids, mask in zip(embs, input_ids,attention_masks):
                            docid += 1
                            maskedIds = ids[(mask > 0) & (ids != 0)]
                            lenId = len(maskedIds)
                            lenEmb = len(emb)
                            if lenId != lenEmb:
                                print("docid %d lenMaskedIds %d lenEmb %d" % (docid, lenId, lenEmb) )
                                print(ids)
                                print(maskedIds)
                                print(mask)                    
                                print(docs[reverse_indices[docid]])
                                print(emb)
                                assert False

                #print("len D_i = %d" % len(D_i))
                left = [D[idx] for idx in reverse_indices.tolist()]
                right = [D_i[idx] for idx in reverse_indices.tolist()]
                
                lexical_left = [lexical_D[idx] for idx in reverse_indices.tolist()]

                return left, lexical_left, right

            return [D[idx] for idx in reverse_indices.tolist()], [lexical_D[idx] for idx in reverse_indices.tolist()]
            """

        input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
        if with_ids:
            # we remove any tokenids that are in the skiplist (by setting them to 0, the padding id), and we move them to the end
            rtr_ids = input_ids.clone()
            if self.skiplist.shape[0] > 9:
                rtr_ids[torch.isin(input_ids, self.skiplist)] = 0
                rtr_ids = rtr_ids[torch.arange(rtr_ids.shape[0]).unsqueeze(1), (rtr_ids == 0).sort(dim=1, stable=True).indices]
            return self.doc(input_ids, attention_mask, keep_dims=keep_dims), rtr_ids
        return self.doc(input_ids, attention_mask, keep_dims=keep_dims)

    def score(self, Q, D, mask=None, lengths=None, explain=False):
        assert False # not used in IsoBERT
        if lengths is not None:
            assert mask is None, "don't supply both mask and lengths"

            mask = torch.arange(D.size(1), device=DEVICE) + 1
            mask = mask.unsqueeze(0) <= lengths.to(DEVICE).unsqueeze(-1)

        scores = (D @ Q)
        scores = scores if mask is None else scores * mask.unsqueeze(-1)
        scores = scores.max(1)

        if explain:
            assert False, "TODO"

        return scores.values.sum(-1).cpu()


def _stack_3D_tensors(groups):
    bsize = sum([x.size(0) for x in groups])
    maxlen = max([x.size(1) for x in groups])
    hdim = groups[0].size(2)

    output = torch.zeros(bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype)

    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, :x.size(1)] = x
        offset = endpos

    return output
