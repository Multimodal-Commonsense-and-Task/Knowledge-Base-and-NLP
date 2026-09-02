
'''
Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
This software is licensed under the BSD 3-Clause License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
PURPOSE.
'''

import os
import shutil
from torch.optim import Adam
from jiwer import wer
import argparse
from phoneme_data_reader import *
from phoneme_model import *
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

def train_epoch(model, train_data_loader, optimizer,args):
    loss_list = []
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    model.train()
    for i, batch in enumerate(train_data_loader):
        optimizer.zero_grad()
        tag_loss, gen_loss,phoneme_gen_loss,  total_loss = model(batch)
        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            tag_loss = tag_loss.mean()
            gen_loss = gen_loss.mean()
            phoneme_gen_loss =phoneme_gen_loss.mean()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_list.append(total_loss)
        if i % 20 == 0:  # monitoring
            if args.local_rank in [-1, 0]:
                print(f"train step: {i}, tag loss is {tag_loss.item()}, gen loss is {gen_loss.item()},  phoneme gen loss is {phoneme_gen_loss.item()}, total loss is {total_loss.item()}")

    return sum(loss_list) / len(loss_list)

def valid_epoch(model, valid_data_loader, args):
    if args.n_gpu > 1:
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)
    model.eval()
    losses = []
    tag_losses = []
    gen_losses = []
    phoneme_gen_losses = []
    with torch.no_grad():
        for i, batch in enumerate(valid_data_loader):
            tag_loss, gen_loss,phoneme_gen_loss, loss = model(batch)
            #print(tag_loss)
            #print(gen_loss)
            #print(loss)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                tag_loss = tag_loss.mean()
                gen_loss = gen_loss.mean()
                phoneme_gen_loss = phoneme_gen_loss.mean()
            #print(args.device, loss)
            #print(args.device, tag_loss)
            #print(args.device, gen_loss)
            losses += [loss.item()]
            tag_losses += [tag_loss.item()]
            gen_losses += [gen_loss.item()]
            phoneme_gen_losses += [phoneme_gen_loss.item()]
    return sum(losses)/len(losses), sum(tag_losses)/len(tag_losses), sum(gen_losses)/len(gen_losses), sum(phoneme_gen_losses)/len(phoneme_gen_losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    #model
    parser.add_argument("--base_model", type=str, default="", help="")
    parser.add_argument("--tag_pdrop", type=float, default=0.2, help="")
    parser.add_argument("--decoder_proj_pdrop", type=float, default=0.2, help="")
    parser.add_argument("--tag_hidden_size", type=int, default=768, help="")
    parser.add_argument("--tag_size", type=int, default=3, help="")
    parser.add_argument("--vocab_size", type=int, default=30522, help="")
    parser.add_argument("--pad_token_id", type=int, default=0, help="")
    parser.add_argument("--alpha", type=float, default=3.0, help="")
    parser.add_argument("--beta", type=float, default=0.5, help="")
    parser.add_argument("--change_weight", type=float, default=1.5, help="")

    #data
    parser.add_argument("--train_data_file", type=str, default="", help="")
    parser.add_argument("--eval_data_file", type=str, default="", help="")
    parser.add_argument("--max_src_len", type=int, default=256, help="")
    parser.add_argument("--max_add_len", type=int, default=10, help="")
    parser.add_argument("--tokenizer_name", type=str, default="", help="")

    #train
    parser.add_argument("--batch_size", type=int, default=32, help="")
    parser.add_argument("--softlabel_path", type=str, default="phoneme_softlabel_matrix.pkl", help="")
    parser.add_argument("--lr", type=float, default=5e-5, help="")
    parser.add_argument("--max_num_epochs", type=int, default=10, help="")
    parser.add_argument("--save_dir", type=str, default="", help="")
    parser.add_argument("--device", type=str, default="", help="")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")

    args = parser.parse_args()
    args.distributed = (args.local_rank != -1)
    if not args.distributed:
        #args.device = "cuda:" + args.device
        #print(args.device)
        device=torch.device("cuda", 0)
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    args.n_gpu = torch.cuda.device_count() if not args.distributed else 1
    args.device = device
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab
    #define model, loss_fn, optimizer
    model = TagDecoder(args)
    model = model.to(args.device)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    #load data
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name, \
                                              do_lower_case=True, do_basic_tokenize=False)
    train_examples = get_examples(examples_path=args.train_data_file,
                                  tokenizer=tokenizer,
                                  max_src_len=args.max_src_len,
                                  max_add_len=args.max_add_len,
                                  )
    eval_examples = get_examples(examples_path=args.eval_data_file,
                                 tokenizer=tokenizer,
                                 max_src_len=args.max_src_len,
                                 max_add_len=args.max_add_len,
                                 )

    train_dataset = ExampleDataset_phoneme(train_examples, args.softlabel_path)
    valid_dataset = ExampleDataset_phoneme(eval_examples, args.softlabel_path)
    train_data_loader = DataLoader(train_dataset, collate_fn=collate_batch_phoneme, batch_size=args.batch_size, shuffle=True) if args.local_rank == -1 else DataLoader(train_dataset, collate_fn=collate_batch_phoneme, 
    sampler = DistributedSampler(train_dataset),
    batch_size=args.batch_size)
    valid_data_loader = DataLoader(valid_dataset, collate_fn=collate_batch_phoneme,batch_size=args.batch_size, shuffle = False) if args.local_rank ==-1 else DataLoader(valid_dataset, collate_fn=collate_batch_phoneme, 
    sampler = DistributedSampler(valid_dataset, shuffle = False),
    batch_size=args.batch_size)


    

    optimizer = Adam(model.parameters(), lr=args.lr)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    eval_loss_list = []
    for epoch in range(1, args.max_num_epochs + 1):
        if args.local_rank in [-1, 0]:
            print(f"=========train at epoch={epoch}=========")
        avg_train_loss = train_epoch(model, train_data_loader, optimizer,args)
        if args.local_rank in [-1, 0]:
            print(f"train {epoch} average loss is {avg_train_loss}")
        if args.local_rank in [-1, 0]:
            print(f"=========eval at epoch={epoch}=========")
        avg_val_loss, avg_val_tag_loss, avg_val_gen_loss, avg_val_phoneme_gen_loss = valid_epoch(model, valid_data_loader,args)
        if args.local_rank in [-1, 0]:
            print(f"eval {epoch} tag loss is {avg_val_tag_loss}, gen loss is {avg_val_gen_loss}, phoneme gen loss is {avg_val_phoneme_gen_loss}, total loss is {avg_val_loss}")
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel traini
            torch.save(model_to_save.state_dict(), args.save_dir + f"/{epoch}.pt")
            eval_loss_list.append((epoch, avg_val_loss))

    eval_loss_list.sort(key=lambda x:x[-1])
    if args.local_rank in [-1, 0]:
        print(eval_loss_list)
        best_epoch_path = os.path.join(args.save_dir, str(eval_loss_list[0][0]) + ".pt")
        print(f"best epoch path is {best_epoch_path}")
        shutil.copyfile(best_epoch_path, os.path.join(args.save_dir, f"best.pt"))
