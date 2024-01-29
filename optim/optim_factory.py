from torch import optim


def create_optimizer(config, model):
    opt_lower = config['opt'].lower()
    parameters = model.parameters()

    if opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, config['lr'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    return optimizer