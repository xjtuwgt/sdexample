from toyexample_experiments import default_argparser, complete_default_parser
from toyexample_experiments import train_data_loader, dev_data_loader, test_data_loader
from toyexample_experiments import seed_everything, model_builder, model_evaluation, save_match_model
import logging
import torch
from envs import OUTPUT_FOLDER
from tqdm import tqdm, trange
from os.path import join


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = default_argparser()
args = parser.parse_args()
for key, value in vars(args).items():
    print('{}\t{}'.format(key, value))
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
args = complete_default_parser(args=args)
train_dataloader = train_data_loader(args=args)
dev_dataloader = dev_data_loader(args=args)
test_dataloader = test_data_loader(args=args)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# torch.cuda.manual_seed_all(args.seed)
seed_everything(seed=args.seed)
model = model_builder(args=args)
model = model.to(args.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
step = 0
start_epoch = 0
best_dev_acc = -1
test_acc = -1
best_step = None
window_step = 0
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
train_iterator = trange(start_epoch, start_epoch + int(args.epochs), desc="Epoch")
for epoch in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    train_correct = 0
    train_total = 0
    for batch_idx, batch in enumerate(epoch_iterator):
        model.train()
        batch = {k: batch[k].to(args.device) for k in batch}
        input = batch['input'].clamp(min=0)
        attn_mask = (input >= 0)
        loss, logits = model(input, attention_mask=attn_mask, labels=batch['labels'])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        pred = logits.max(1)[1]
        train_total = train_total + pred.shape[0]
        train_correct += (pred == batch['labels']).sum()
        if (step + 1) % args.eval_batch_interval_num == 0:
            dev_acc = model_evaluation(model=model, data_loader=dev_dataloader, args=args)
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                test_acc = model_evaluation(model=model, data_loader=test_dataloader, args=args)
                if args.save_model:
                    model_name = join(OUTPUT_FOLDER, 'model_dev_{:.4f}_test_{:.4f}.pkl'.format(dev_acc, test_acc))
                    save_match_model(model=model, model_name=model_name)
                best_step = (epoch + 1, step + 1)
                window_step = 0
            else:
                window_step = window_step + 1
            logging.info("Step {}: dev accuracy={:.6f}, current best dev accuracy={:.6f} and test accuracy = {:.6f}".format((epoch + 1, step + 1), dev_acc, best_dev_acc, test_acc), flush=True)
            if window_step >= args.window_size:
                break
        if window_step >= args.window_size:
            break
        step = step + 1
    logging.info('Train accuracy = {:.6f} at {}'.format(train_correct *1.0 /train_total, epoch))
    if window_step >= args.window_size:
        break
logging.info("Best dev result at {} dev accuracy={:.6f} test accuracy = {:.6f}".format(best_step, best_dev_acc, test_acc))
logging.info('*'*25)
for key, value in vars(args).items():
    logging.info('{}\t{}'.format(key, value))
logging.info('*' * 25)