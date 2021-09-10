from data_utils.findcat import FindCatDataset, find_cat_validation_fn, find_cat_collate_fn
from data_utils.dataset import SentenceDropDataset
from torch.utils.data import DataLoader
from data_utils.findcat import MASK
from data_utils.model_prober import ProberModel
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
from data_utils.model_prober import DROP_PROBE_MODEL_NAME, ORIG_PROBE_MODEL_NAME
########################################################################################################################

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

if __name__ == '__main__':
    print()
    parser = prober_default_parser()
    args = parser.parse_args()
    args = complete_default_parser(args=args)
    args.pre_trained_file_name = None
    for key, value in vars(args).items():
        print('{}\t{}'.format(key, value))

    model = ProberModel(config=args)
    model.bert.register_forward_hook(get_activation('encoder'))
    print(model)


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