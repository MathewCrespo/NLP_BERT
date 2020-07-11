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