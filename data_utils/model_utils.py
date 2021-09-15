import torch
from tqdm import tqdm
import transformers

def model_evaluation(model, data_loader, args):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = {k: batch[k].to(args.device) for k in batch}
            input = batch['input'].clamp(min=0)
            attn_mask = (input >= 0)
            _, logits = model(input, attention_mask=attn_mask, labels=batch['labels'])
            pred = logits.max(1)[1]
            total += len(pred)
            correct += (pred == batch['labels']).sum()
            del batch
    acc = correct * 1.0 / total
    return acc

def model_loss_computation(model, data_loader, args):
    model.eval()
    total = 0
    correct = 0
    loss_list = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = {k: batch[k].to(args.device) for k in batch}
            input = batch['input'].clamp(min=0)
            attn_mask = (input >= 0)
            loss, logits = model(input, attention_mask=attn_mask, labels=batch['labels'])
            loss_list.append(loss.detach().item())
            pred = logits.max(1)[1]
            total += len(pred)
            correct += (pred == batch['labels']).sum()
            del batch
    dev_acc = correct * 1.0 / total
    avg_loss = sum(loss_list)/len(loss_list)
    return avg_loss


def model_builder(args):
    model_config = transformers.AutoConfig.from_pretrained(args.model_name)
    model_config.num_hidden_layers = 3
    model_config.vocab_size = args.vocab_size
    model = transformers.AutoModelForSequenceClassification.from_config(model_config)
    # print('Model Parameter Configuration:')
    # for name, param in model.named_parameters():
    #     print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
    # print('*' * 75)
    return model

def load_pretrained_model(args, pretrained_model_name):
    model_config = transformers.AutoConfig.from_pretrained(args.model_name)
    model_config.num_hidden_layers = 3
    model_config.vocab_size = args.vocab_size
    model = transformers.AutoModelForSequenceClassification.from_config(model_config)
    model.load_state_dict(torch.load(pretrained_model_name))
    print('Loading model from {}'.format(pretrained_model_name))
    return model