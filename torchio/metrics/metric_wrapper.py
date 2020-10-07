from ..torchio import DATA
from .base_metric import Metric
import torch
from typing import Callable, Union, List


class MetricWrapper(Metric):

    def __init__(self, metric_name: str, metric_func: Callable, use_mask: bool = False, mask_key: str = None,
                 select_key: Union[List, str] = None, scale_metric: float = 1.0, save_in_subject_keys: bool = False):
        super(MetricWrapper, self).__init__(metric_name=metric_name, select_key=select_key, scale_metric=scale_metric,
                                            save_in_subject_keys=save_in_subject_keys)
        self.metric_func = metric_func
        self.use_mask = use_mask
        self.mask_key = mask_key
        self.select_key = select_key

    def apply_metric(self, sample1, sample2):
        common_keys = self.get_common_intensity_keys(sample1=sample1, sample2=sample2)
        computed_metrics = {}
        for sample_key in common_keys:
            if sample_key is self.mask_key:
                continue
            data1 = sample1[sample_key][DATA]
            data2 = sample2[sample_key][DATA]

            if self.use_mask and self.mask_key is not None:
                mask_data1, mask_data2 = sample1[self.mask_key][DATA], sample2[self.mask_key][DATA]
                data1 = torch.mul(data1, mask_data1)
                data2 = torch.mul(data2, mask_data2)
            computed_metrics[sample_key] = dict()
            #Compute metric
            result = self.metric_func(data1, data2)
            if isinstance(result, dict):
                for key_metric, value_metric in result.items():
                    computed_metrics[sample_key][self.metric_name + "_" + key_metric] = value_metric
            else:
                computed_metrics[sample_key][self.metric_name] = result
        return computed_metrics
