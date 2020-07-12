import os
from gluonnlp.data.dataset import TSVDataset
from gluonnlp.data.registry import register


class _CNSADataset(TSVDataset):
    def __init__(self, root, dataset_name, segment, **kwargs):
        self._root = root
        filename = os.path.join(self._root, dataset_name, 'waimai_%s.tsv' % segment)
        super(_CNSADataset, self).__init__(filename, **kwargs)


@register(segment=['train', 'dev', 'test'])
class CNSAData(_CNSADataset):

    def __init__(self, segment='train',
                 root=os.path.join(os.path.dirname(os.path.abspath(__file__))),
                 return_all_fields=False):
        A_IDX, LABEL_IDX = 0, 1
        if segment in ['train', 'dev']:
            field_indices = [A_IDX, LABEL_IDX] if not return_all_fields else None
            num_discard_samples = 1
        elif segment == 'test':
            field_indices = [A_IDX] if not return_all_fields else None
            num_discard_samples = 1

        super(CNSAData, self).__init__(root, 'waimai', segment,
                                             num_discard_samples=num_discard_samples,
                                             field_indices=field_indices)


class _WeiboDataset(TSVDataset):
    def __init__(self, root, dataset_name, segment, **kwargs):
        self._root = root
        filename = os.path.join(self._root, dataset_name, 'weibo_%s.tsv' % segment)
        super(_WeiboDataset, self).__init__(filename, **kwargs)


@register(segment=['train', 'dev', 'test'])
class WeiboData(_WeiboDataset):

    def __init__(self, segment='train',
                 root=os.path.join(os.path.dirname(os.path.abspath(__file__))),
                 return_all_fields=False):
        A_IDX, LABEL_IDX = 0, 1
        if segment in ['train', 'dev']:
            field_indices = [A_IDX, LABEL_IDX] if not return_all_fields else None
            num_discard_samples = 1
        elif segment == 'test':
            field_indices = [A_IDX] if not return_all_fields else None
            num_discard_samples = 1

        super(WeiboData, self).__init__(root, 'weibo', segment,
                                             num_discard_samples=num_discard_samples,
                                             field_indices=field_indices)

class _Weibo2Dataset(TSVDataset):
    def __init__(self, root, dataset_name, segment, **kwargs):
        self._root = root
        filename = os.path.join(self._root, dataset_name, 'weibo2_%s.tsv' % segment)
        super(_Weibo2Dataset, self).__init__(filename, **kwargs)


@register(segment=['train', 'dev', 'test'])
class Weibo2Data(_Weibo2Dataset):

    def __init__(self, segment='train',
                 root=os.path.join(os.path.dirname(os.path.abspath(__file__))),
                 return_all_fields=False):
        A_IDX, LABEL_IDX = 0, 1
        if segment in ['train', 'dev']:
            field_indices = [A_IDX, LABEL_IDX] if not return_all_fields else None
            num_discard_samples = 1
        elif segment == 'test':
            field_indices = [A_IDX] if not return_all_fields else None
            num_discard_samples = 1

        super(Weibo2Data, self).__init__(root, 'weibo2', segment,
                                             num_discard_samples=num_discard_samples,
                                             field_indices=field_indices)

@register(segment=['train', 'dev', 'test'])
class _LCQMCDataset(TSVDataset):
    def __init__(self, root, dataset_name, segment, **kwargs):
        self._root = root
        filename = os.path.join(self._root, dataset_name, 'lcqmc_%s.tsv' % segment)
        super(_LCQMCDataset, self).__init__(filename, **kwargs)
        
class LCQMC(_LCQMCDataset):
    """ The LCQMC dataset redistributed by Baidu
    <https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE>.

    Original from:
    Xin Liu, Qingcai Chen, Chong Deng, Huajun Zeng, Jing Chen, Dongfang Li, Buzhou Tang,
        LCQMC: A Large-scale Chinese Question Matching Corpus,COLING2018.
    Licensed under a Creative Commons Attribution 4.0 International License. License details:
        http://creativecommons.org/licenses/by/4.0/

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/baidu_ernie_task_data'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    return_all_fields : bool, default False
        Return all fields available in the dataset.

    Examples
    --------
    >>> lcqmc_dev = BaiduErnieLCQMC('dev', root='./datasets/baidu_ernie_task_data/')
    -etc-
    >>> len(lcqmc_dev)
    8802
    >>> len(lcqmc_dev[0])
    3
    >>> lcqmc_dev[0]
    ['开初婚未育证明怎么弄？', '初婚未育情况证明怎么开？', '1']
    >>> lcqmc_test = BaiduErnieLCQMC('test', root='./datasets/baidu_ernie_task_data/')
    -etc-
    >>> len(lcqmc_test)
    12500
    >>> len(lcqmc_test[0])
    2
    >>> lcqmc_test[0]
    ['谁有狂三这张高清的', '这张高清图，谁有']
    """
    def __init__(self, segment='train',
                 root=os.path.join(os.path.dirname(os.path.abspath(__file__))),
                 return_all_fields=False):
        A_IDX, B_IDX, LABEL_IDX = 0, 1, 2
        if segment in ['train', 'dev']:
            field_indices = [A_IDX, B_IDX, LABEL_IDX] if not return_all_fields else None
            num_discard_samples = 1
        elif segment == 'test':
            field_indices = [A_IDX, B_IDX] if not return_all_fields else None
            num_discard_samples = 1

        super(LCQMC, self).__init__(root, 'lcqmc', segment,
                                              num_discard_samples=num_discard_samples,
                                              field_indices=field_indices)