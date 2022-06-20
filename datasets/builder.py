import logging
from datasets import mnms,scgm


logger = logging.getLogger("global")

def get_loader(dataset='mnms',img_size=288,batch_size=1,domain='A'):
    if dataset == 'mnms':
        train_loader_sup,train_loader_unsup,val_loader \
            = mnms.get_split_data(target_vendor=domain,image_size=img_size,batch_size=batch_size)
        logger.info('Get loader Done...')
        return train_loader_sup,train_loader_unsup,val_loader

    elif dataset == 'scgm':
        train_loader_sup,train_loader_unsup,val_loader \
            = scgm.get_split_data(target_vendor=domain,image_size=img_size,batch_size=batch_size)
        return train_loader_sup, train_loader_unsup,val_loader

    else:
        raise NotImplementedError(
            "dataset type {} is not supported".format(dataset)
        )
