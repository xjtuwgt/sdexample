from data_utils.findcat import FindCatDataset, find_cat_validation_fn, find_cat_collate_fn
from data_utils.dataset import SentenceDropDataset
from torch.utils.data import DataLoader
from data_utils.findcat import MASK
from data_utils.model_prober import ProberModel
from data_utils.model_utils import load_pretrained_model, model_builder
from data_utils.argument_parser import prober_default_parser, complete_default_parser
from data_utils.model_prober import ProberModel
import torch
import os
from envs import OUTPUT_FOLDER, HOME_DATA_FOLDER
from tqdm import tqdm, trange
from toyexample_experiments import seed_everything
from os.path import join
from data_utils.findcat import FindCatDataset, find_cat_probe_collate_fn
from torch.utils.data import DataLoader
from data_utils.model_prober import probe_model_evaluation
from data_utils.model_prober import DROP_PROBE_MODEL_NAME, ORIG_PROBE_MODEL_NAME
from envs import OUTPUT_FOLDER
########################################################################################################################

# activation = {}
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output
#     return hook

def list_all_folders(d, model_type: str):
    folder_names = [os.path.join(d, o) for o in os.listdir(d)
     if os.path.isdir(os.path.join(d, o))]
    folder_names = [i for i in folder_names if model_type in i]
    return folder_names

def list_all_extension_files(path, extension='.log'):
    files = os.listdir(path)
    files = [i for i in files if i.endswith(extension)]
    return files

if __name__ == '__main__':
    print()
    ORIG_MODEL = 'model_False_0.0'
    DROP_MODEL = 'model_False_0.1'
    BETA_DROP_MODEL = 'model_True_0.0'
    folder_names = list_all_folders(d=OUTPUT_FOLDER, model_type='.models')
    for folder in folder_names:
        print(folder)
        best_orig_metric = 0.0
        best_orig_model = ''
        best_drop_metric = 0.0
        best_drop_model = ''
        best_beta_metric = 0.0
        best_beta_model = ''
        model_names = list_all_extension_files(path=folder, extension='.pkl')
        for model_name in model_names:
            print(model_name)
            start_idx = model_name.rindex('_')
            end_idx = model_name.rindex('.')
            model_metric = model_name[(start_idx + 1):end_idx]
            print(model_metric)
            # if model_name.startswith(ORIG_MODEL):


        print('*' * 50)
    # parser = prober_default_parser()
    # args = parser.parse_args()
    # args = complete_default_parser(args=args)
    # args.pre_trained_file_name = None
    # for key, value in vars(args).items():
    #     print('{}\t{}'.format(key, value))
    #
    # model = ProberModel(config=args)
    # model.bert.register_forward_hook(get_activation('encoder'))
    # print(model)


    # cat_data_set = FindCatDataset(total_examples=5)
    # sent_dropout = 0.1
    # batch_size = 1
    # validate_examples = False
    # mask = True
    # validation_fn = find_cat_validation_fn if validate_examples else lambda ex: True
    #
    # sdrop_dataset = SentenceDropDataset(cat_data_set, sent_drop_prob=sent_dropout,
    #     example_validate_fn=validation_fn, mask=mask, mask_id=MASK)
    #
    # dataloader = DataLoader(sdrop_dataset, batch_size=batch_size, collate_fn=find_cat_collate_fn)
    #
    # for batch_idx, batch in enumerate(dataloader):
    #     print(batch['input'].shape)