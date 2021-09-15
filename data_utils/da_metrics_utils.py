###Tradeoffs in Data Augmentation: An Empirical Study, https://openreview.net/pdf?id=ZcKPWuhG6wy
from data_utils.model_utils import model_loss_computation, model_evaluation

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

