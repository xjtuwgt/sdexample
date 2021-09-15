from data_utils.argument_parser import data_aug_default_parser, complete_default_parser
from toyexample_experiments import orig_da_train_data_loader, orig_da_dev_data_loader, drop_da_train_data_loader, drop_da_dev_data_loader
from toyexample_experiments import seed_everything
from data_utils.da_metrics_utils import affinity_metrics_computation, diversity_metrics_computation
import torch
from data_utils.model_utils import load_pretrained_model
from envs import OUTPUT_FOLDER
from tqdm import tqdm, trange
from os.path import join
import os

parser = data_aug_default_parser()
args = parser.parse_args()
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
args = complete_default_parser(args=args)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if args.exp_name is None:
    args.exp_name = args.train_file_name + '.models'
    args.exp_name = join(OUTPUT_FOLDER, args.exp_name)
    os.makedirs(args.exp_name, exist_ok=True)
for key, value in vars(args).items():
    print('{}\t{}'.format(key, value))
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
orig_train_dataloader = orig_da_train_data_loader(args=args)
orig_dev_dataloader = orig_da_dev_data_loader(args=args)

drop_train_dataloader = drop_da_train_data_loader(args=args)
drop_dev_data_loader = drop_da_dev_data_loader(args=args)
# test_dataloader = test_data_loader(args=args)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# torch.cuda.manual_seed_all(args.seed)
seed_everything(seed=args.seed)
if args.orig_model_name is not None:
    orig_model = load_pretrained_model(args=args, pretrained_model_name=args.orig_model_name)
    affinity_metrics = affinity_metrics_computation(model=orig_model, dev_data_loader=orig_dev_dataloader,
                                                    drop_dev_data_loader=drop_dev_data_loader, args=args)
else:
    affinity_metrics = 0.0

if args.drop_model_name is not None and args.orig_model_name is not None:
    drop_model = load_pretrained_model(args=args, pretrained_model_name=args.orig_model_name)
    orig_model = load_pretrained_model(args=args, pretrained_model_name=args.orig_model_name)
    diversity_metrics = diversity_metrics_computation(model=orig_model, drop_model=drop_model,
                                                      train_data_loader=orig_train_dataloader,
                                                      drop_train_data_loader=drop_train_dataloader,
                                                      args=args)
else:
    diversity_metrics = 0.0

for key, value in vars(args).items():
    print(key, value)

print('Affinity metrics: {}'.format(affinity_metrics))
print('Diversity metrics: {}'.format(diversity_metrics))

# model = model_builder(args=args)
# model = model.to(args.device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# step = 0
# start_epoch = 0
# best_dev_acc = -1
# test_acc = -1
# best_step = None
# window_step = 0
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# train_iterator = trange(start_epoch, start_epoch + int(args.epochs), desc="Epoch")
# for epoch_idx, epoch in enumerate(train_iterator):
#     epoch_iterator = tqdm(train_dataloader, desc="Iteration")
#     train_correct = 0
#     train_total = 0
#     for batch_idx, batch in enumerate(epoch_iterator):
#         model.train()
#         batch = {k: batch[k].to(args.device) for k in batch}
#         input = batch['input'].clamp(min=0)
#         attn_mask = (input >= 0)
#         loss, logits = model(input, attention_mask=attn_mask, labels=batch['labels'])
#
#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#         pred = logits.max(1)[1]
#         train_total = train_total + pred.shape[0]
#         train_correct += (pred == batch['labels']).sum()
#         if (step + 1) % args.eval_batch_interval_num == 0:
#             dev_acc = model_evaluation(model=model, data_loader=dev_dataloader, args=args)
#             if dev_acc > best_dev_acc:
#                 best_dev_acc = dev_acc
#                 # test_acc = model_evaluation(model=model, data_loader=test_dataloader, args=args)
#                 if args.save_model:
#                     model_name = join(args.exp_name, 'model_{}_{}_{}_{}_dev_{:.4f}.pkl'.format(args.beta_drop, args.sent_dropout,
#                                                                                                epoch_idx+1, batch_idx+1, dev_acc))
#                     save_match_model(model=model, model_name=model_name)
#                 best_step = (epoch + 1, step + 1)
#                 window_step = 0
#             else:
#                 window_step = window_step + 1
#             print("Step {}: dev accuracy={:.6f}, current best dev accuracy={:.6f} and test accuracy = {:.6f}".format((epoch + 1, step + 1), dev_acc, best_dev_acc, test_acc), flush=True)
#             if window_step >= args.window_size:
#                 break
#         if window_step >= args.window_size:
#             break
#         step = step + 1
#     print('Train accuracy = {:.6f} at {}'.format(train_correct *1.0 /train_total, epoch))
#     if window_step >= args.window_size:
#         break
# print("Best dev result at {} dev accuracy={:.6f} test accuracy = {:.6f}".format(best_step, best_dev_acc, test_acc))
# print('*'*25)
# for key, value in vars(args).items():
#     print('{}\t{}'.format(key, value))
# print('*' * 25)