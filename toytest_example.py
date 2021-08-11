from data_utils.findcat import FindCatDataset, find_cat_validation_fn, find_cat_collate_fn
from data_utils.dataset import SentenceDropDataset
from torch.utils.data import DataLoader
from data_utils.findcat import MASK


if __name__ == '__main__':
    cat_data_set = FindCatDataset(total_examples=5)
    sent_dropout = 0.1
    batch_size = 1
    validate_examples = False
    mask = True
    validation_fn = find_cat_validation_fn if validate_examples else lambda ex: True

    sdrop_dataset = SentenceDropDataset(cat_data_set, sent_drop_prob=sent_dropout,
        example_validate_fn=validation_fn, mask=mask, mask_id=MASK)

    dataloader = DataLoader(sdrop_dataset, batch_size=batch_size, collate_fn=find_cat_collate_fn)

    for batch_idx, batch in enumerate(dataloader):
        print(batch['input'].shape)