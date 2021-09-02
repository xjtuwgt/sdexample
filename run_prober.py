from data_utils.model_utils import load_pretrained_model, model_builder
from data_utils.model_prober import prober_default_parser
from data_utils.model_prober import ProberModel
import torch
########################################################################################################################
parser = prober_default_parser()
args = parser.parse_args()
for key, value in vars(args).items():
    print('{}\t{}'.format(key, value))
########################################################################################################################
model = ProberModel(config=args)
input = torch.LongTensor([1,2,3,5]).view(1, -1)
attn_mask = (input >= 0)
x = model(input, attn_mask)
print(x.shape, x)

# print(type(activation['bert']))
# for _ in activation['bert']:
#     print(_.shape)
# print(x)