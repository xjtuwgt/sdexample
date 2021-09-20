###Tradeoffs in Data Augmentation: An Empirical Study, https://openreview.net/pdf?id=ZcKPWuhG6wy
from data_utils.model_utils import model_loss_computation, model_evaluation

MODEL_NAMES = [('train_fastsingle_cat_100_42_300_0.5.pkl.gz.models', {'orig': 'model_False_0.0_50_2_dev_0.5585.pkl',
                                                                      'drop': 'model_False_0.1_800_2_dev_0.5948.pkl',
                                                                      'beta_drop': 'model_True_0.1_800_2_dev_0.6133.pkl'}),
               ('train_fastsingle_cat_200_42_300_0.5.pkl.gz.models', {'orig': 'model_False_0.0_75_4_dev_0.5624.pkl',
                                                                      'drop': '',
                                                                      'beta_drop': ''}),
               ('train_fastsingle_cat_500_42_300_0.5.pkl.gz.models', {'orig': '',
                                                                      'drop': '',
                                                                      'beta_drop': ''}),
               ('train_fastsingle_cat_1000_42_300_0.5.pkl.gz.models', {'orig': '',
                                                                       'drop': '',
                                                                       'beta_drop': ''}),
               ('train_fastsingle_cat_2000_42_300_0.5.pkl.gz.models', {'orig': '',
                                                                       'drop': '',
                                                                       'beta_drop': ''}),
               ('train_fastsingle_cat_5000_42_300_0.5.pkl.gz.models', {'orig': '',
                                                                       'drop': '',
                                                                       'beta_drop': ''}),
               ('train_fastsingle_cat_10000_42_300_0.5.pkl.gz.models', {'orig': '',
                                                                        'drop': '',
                                                                        'beta_drop': ''}),
               ('train_fastsingle_cat_20000_42_300_0.5.pkl.gz.models', {'orig': '',
                                                                        'drop': '',
                                                                        'beta_drop': ''})]

def affinity_metrics_computation(model, dev_data_loader, drop_dev_data_loader, args):
    """
    :param model: train over clean data
    :param dev_data_loader:
    :param drop_dev_data_loader:
    :return:
    Affinity: Acc(model, drop_dev) / Acc(model, dev)
    """
    model = model.to(args.device)
    drop_acc = model_evaluation(model=model, data_loader=drop_dev_data_loader, args=args)
    acc = model_evaluation(model=model, data_loader=dev_data_loader, args=args)
    affinity = drop_acc/acc
    return affinity

def diversity_metrics_computation(model, train_data_loader, drop_model, drop_train_data_loader, args):
    """
    :param model: model trained over clean data
    :param train_data_loader:
    :param drop_model: model trained over augmentation data
    :param drop_train_data_loader:
    :return:
    Diversity: Loss(drop_model, drop_train) / Loss(model, train)
    """
    drop_model = drop_model.to(args.device)
    drop_loss = model_loss_computation(model=drop_model, data_loader=drop_train_data_loader, args=args)
    model = model.to(args.device)
    loss = model_loss_computation(model=model, data_loader=train_data_loader, args=args)
    diversity  = drop_loss / loss
    return diversity

