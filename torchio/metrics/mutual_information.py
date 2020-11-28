import torch
import numpy as np
from torch.nn.functional import pad
from .map_metric_wrapper import MapMetricWrapper
from .utils import spatial_filter_nd, gauss_kernel_3d


class NMI(MapMetricWrapper):
    def __init__(self, metric_name: str = "NMI", metric_func: Callable = nmi_loss,  select_key: Union[List, str] = None, scale_metric: float = 1,
                 average_method: str = None, save_in_subject_keys: bool = False, metric_kwargs: dict = None,
                 **kwargs):
        super(NMI, self).__init__(metric_name=metric_name, metric_func=metric_func, select_key=select_key,
                                  scale_metric=scale_metric, average_method=average_method,
                                  save_in_subject_keys=save_in_subject_keys, metric_kwargs=metric_kwargs, **kwargs)


def nmi_loss(x, y, num_bins=32, patch_size=12, sigma=.02, eps=1e-8):
    #Padding for patch_size
    x_pad = np.asarray(x.shape) % patch_size
    x_pad = [x_pad[j//2] if j % 2 else 0 for j in range(len(x_pad) * 2)]

    y_pad = np.asarray(y.shape) % patch_size
    y_pad = [y_pad[j // 2] if j % 2 else 0 for j in range(len(y_pad) * 2)]
    padded_x, padded_y = pad(x, x_pad), pad(y, y_pad)
    #Extract patches -> dim = (nb_patch, patch_size_flattened) to
    patch_x = padded_x.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size).unfold(2, patch_size,
                                                                                                  patch_size)
    patch_x = patch_x.reshape((-1, patch_size**3))

    patch_y = padded_y.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size).unfold(2, patch_size,
                                                                                                  patch_size)
    patch_y = patch_y.reshape((-1, patch_size**3))
    #Actual MI computation
    preterm = 1 / (sigma * np.sqrt(2*np.pi))

    x_flat, y_flat = x.reshape((-1)), y.reshape((-1))
    #Get histogram bins
    _, bin_centers_x = np.histogram(x_flat, bins=num_bins-1)
    _, bin_centers_y = np.histogram(y_flat, bins=num_bins-1)

    # i_x_patch dimensions: (num_bins, num_patch, patch_size_flattened)
    # p_x_patch dimensions: (num_bins, num_patch) (averaging to get 1 hist distrib per patch)
    i_x_patch = -(patch_x - bin_centers_x.reshape((-1, 1, 1)))**2
    i_x_patch /= 2 * sigma ** 2
    i_x_patch = preterm * torch.exp(i_x_patch)
    #i_x_patch /= i_x_patch.sum(axis=-1, keepdim=True)
    p_x_patch = i_x_patch.mean(axis=2, keepdim=True)

    i_y_patch = -(patch_y - bin_centers_y.reshape((-1, 1, 1)))**2
    i_y_patch /= 2 * sigma ** 2
    i_y_patch = preterm * torch.exp(i_y_patch)
    #i_y_patch /= i_y_patch.sum(axis=-1, keepdim=True)
    p_y_patch = i_y_patch.mean(axis=2, keepdim=True)

    #p_xy_patch dimensions: (num_patch, num_bins, num_bins)
    #Represents the joint distribution of hist of each pair of patch
    p_xy_patch = torch.matmul(i_x_patch.permute(1, 0, 2), i_y_patch.permute(1, 2, 0))
    p_xy_patch /= (patch_size ** 3)
    #p_x_p_y dimensions: (num_bins, num_bins)
    #Represents the product of each pair of bins probabilities
    p_x_p_y_patch = torch.matmul(p_x_patch.permute(1, 0, 2), p_y_patch.permute(1, 2, 0)) + eps
    #Compute mutual information:
    mutual_information = p_xy_patch * torch.log(p_xy_patch/p_x_p_y_patch + eps)
    mutual_information = mutual_information.sum((1, 2)).mean()
    return -mutual_information
