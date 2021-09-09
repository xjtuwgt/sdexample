from data_utils.model_utils import load_pretrained_model, model_builder
from data_utils.argument_parser import prober_default_parser, complete_default_parser
from data_utils.model_prober import ProberModel
import torch
from envs import OUTPUT_FOLDER, HOME_DATA_FOLDER
from tqdm import tqdm, trange
from toyexample_experiments import seed_everything
from os.path import join
from data_utils.findcat import FindCatDataset, find_cat_probe_collate_fn
from torch.utils.data import DataLoader
from data_utils.model_prober import probe_model_evaluation
########################################################################################################################
parser = prober_default_parser()
args = parser.parse_args()
args = complete_default_parser(args=args)
for key, value in vars(args).items():
    print('{}\t{}'.format(key, value))
########################################################################################################################
def train_data_loader(args):
    train_seq_len = args.train_seq_len
    train_seq_len = [int(_) for _ in train_seq_len.split(',')]
    if args.train_file_name is not None:
        train_file_name = join(HOME_DATA_FOLDER, 'toy_data', args.train_file_name)
    else:
        train_file_name = None
    dataset = FindCatDataset(seed=args.seed,
                             target_tokens=args.target_tokens,
                             seqlen=train_seq_len,
                             total_examples=args.train_examples,
                             multi_target=args.multi_target in ['multi'],
                             data_file_name=train_file_name)
    dataloader = DataLoader(dataset=dataset,
                            shuffle=True,
                            batch_size=args.batch_size,
                            collate_fn=find_cat_probe_collate_fn)
    return dataloader

def dev_data_loader(args):
    dev_seq_len = args.eval_test_seq_len
    dev_seq_len = [int(_) for _ in dev_seq_len.split(',')]
    if args.test_file_name is not None:
        dev_file_name = join(HOME_DATA_FOLDER, 'toy_data', args.eval_file_name)
        print('Dev data file name = {}'.format(dev_file_name))
    else:
        dev_file_name = None
    dataset = FindCatDataset(seed=2345,
                             seqlen=dev_seq_len,
                             total_examples=args.test_examples,
                             multi_target=args.multi_target in ['multi'],
                             data_file_name=dev_file_name)
    dev_dataloader = DataLoader(dataset, batch_size=args.test_batch_size,
                                collate_fn=find_cat_probe_collate_fn)
    return dev_dataloader
########################################################################################################################
model = ProberModel(config=args)
model = model.to(args.device)
print('Model Parameter Configuration:')
for name, param in model.named_parameters():
    print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
print('*' * 75)
########################################################################################################################
seed_everything(seed=args.seed)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
train_dataloader = train_data_loader(args=args)
dev_dataloader = dev_data_loader(args=args)
########################################################################################################################
step = 0
start_epoch = 0
best_dev_f1 = -1
test_f1 = -1
best_metrics = (-1, -1)
best_step = None
window_step = 0
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
train_iterator = trange(start_epoch, start_epoch + int(args.epochs), desc="Epoch")
for epoch_idx, epoch in enumerate(train_iterator):
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    train_correct = 0
    train_total = 0
    for batch_idx, batch in enumerate(epoch_iterator):
        model.train()
        batch = {k: batch[k].to(args.device) for k in batch}
        input = batch['input'].clamp(min=0)
        attn_mask = (input >= 0)
        loss, logits = model(input, attention_mask=attn_mask, labels=batch['seq_labels'], label_mask=batch['seq_mask'])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # pred = logits.max(1)[1]
        # train_total = train_total + pred.shape[0]
        # train_correct += (pred == batch['labels']).sum()
        if (step + 1) % args.eval_batch_interval_num == 0:
            dev_em, dev_f1 = probe_model_evaluation(model=model, data_loader=dev_dataloader, args=args)
            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_metrics = (dev_em, dev_f1)
                # test_acc = model_evaluation(model=model, data_loader=test_dataloader, args=args)
                if args.save_model:
                    model_name = join(OUTPUT_FOLDER, 'model_{}_{}_{}_{}_dev_{:.4f}.pkl'.format(args.beta_drop, args.sent_dropout,
                                                                                               epoch_idx+1, batch_idx+1, dev_f1))
                best_step = (epoch + 1, step + 1)
                window_step = 0
            else:
                window_step = window_step + 1
            print("Step {}: dev f1 ={:.6f}, current best dev f1={:.6f} and dev em = {:.6f}".format((epoch + 1, step + 1), dev_f1, best_dev_f1, best_metrics[0]), flush=True)
            if window_step >= args.window_size:
                break
        if window_step >= args.window_size:
            break
        step = step + 1
        if step % 200 == 0:
            print('Train loss = {:.6f} at {}/{}'.format(loss.item(), epoch, batch_idx))
    if window_step >= args.window_size:
        break
print("Best dev result at {} dev f1 ={:.6f} best em = {:.6f}".format(best_step, best_dev_f1, best_metrics[0]))
print('*'*25)
for key, value in vars(args).items():
    print('{}\t{}'.format(key, value))
print('*' * 25)
# input = torch.LongTensor([1,2,3,5]).view(1, -1)
# attn_mask = (input >= 0)
# x = model(input, attn_mask)
# print(x.shape, x)

# print(type(activation['bert']))
# for _ in activation['bert']:
#     print(_.shape)
# print(x)