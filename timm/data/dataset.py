""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2019, Ross Wightman
"""
import io
import logging
from typing import Optional

import torch
import torch.utils.data as data
from PIL import Image
from dwt import *
from .readers import create_reader
import torch_dwt as tdwt
import torchvision.transforms as T

import cvt_functions as cCVT
_logger = logging.getLogger(__name__)
_ERROR_RETRY = 50


class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            load_bytes=False,
            img_mode='RGB',
            transform=None,
            target_transform=None,
            corrupted=None
    ):
        if reader is None or isinstance(reader, str):
            reader = create_reader(
                reader or '',
                root=root,
                split=split,
                class_map=class_map
            )
        self.reader = reader
        self.load_bytes = load_bytes
        self.img_mode = img_mode
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0
        self.corrupted=corrupted
        
        self.dwt_transform = list()
        self.dwt_level = 0
        
    def __getitem__(self, index):
        img, target = self.reader[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.reader.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.reader))
            else:
                raise e
        self._consecutive_errors = 0
        #print('image type -> ', type(img)) #PIL
        #print('Total Sum -> ', get_dwt_components(img))
        #get_dwt_components(img)
        
        if self.corrupted != None and self.corrupted != 'None':
            assert len(self.corrupted.split('_')) == 2, 'Report Error: You must write corrupted level with _'
            c_name = self.corrupted.split('_')[0]
            c_level = self.corrupted.split('_')[1]
            c_fun = cCVT.get_function(c_name)
            img = img.convert(self.img_mode)
            np_img = c_fun(img, int(c_level))
            img = Image.fromarray(np.uint8(np_img), mode='RGB')
            img = img.convert(self.img_mode)
        else:   
            img = img.convert(self.img_mode)
        
        dwt_ratios = get_dwt_components(img)
        
        LL_SUM = dwt_ratios[0] + dwt_ratios[4] + dwt_ratios[8] + dwt_ratios[12]
        recip = np.reciprocal(LL_SUM)
        LH_SUM = dwt_ratios[1] + dwt_ratios[5] + dwt_ratios[9] + dwt_ratios[13] * recip
        HL_SUM = dwt_ratios[2] + dwt_ratios[6] + dwt_ratios[10] + dwt_ratios[14] * recip
        HH_SUM = dwt_ratios[3] + dwt_ratios[7] + dwt_ratios[11] + dwt_ratios[15] * recip
        
        dwt_ratios = [(LL_SUM * recip).item(), LH_SUM.item(), HL_SUM.item(), HH_SUM.item()]
        
        #if self.img_mode and not self.load_bytes:
        #    img = img.convert(self.img_mode)
        if self.transform is not None:
            img = self.transform(img)
        
        '''
        img = tdwt.get_dwt_pil(img)
        img = T.functional.pil_to_tensor(img)
        
        Here, DWT and Normalization
        1, 2, 3
        
        dwt_lst = list()
        
        for idx, trans_fun in enumerate(self.dwt_transform):
            dwt_lst[idx] = trans_fun(dwt_lst[idx])
        ''' 
        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, [target, dwt_ratios]
        

    def __len__(self):
        return len(self.reader)

    def filename(self, index, basename=False, absolute=False):
        return self.reader.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            is_training=False,
            batch_size=None,
            seed=42,
            repeats=0,
            download=False,
            transform=None,
            target_transform=None,
    ):
        assert reader is not None
        if isinstance(reader, str):
            self.reader = create_reader(
                reader,
                root=root,
                split=split,
                is_training=is_training,
                batch_size=batch_size,
                seed=seed,
                repeats=repeats,
                download=download,
            )
        else:
            self.reader = reader
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.reader:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            yield img, target

    def __len__(self):
        if hasattr(self.reader, '__len__'):
            return len(self.reader)
        else:
            return 0

    def set_epoch(self, count):
        # TFDS and WDS need external epoch count for deterministic cross process shuffle
        if hasattr(self.reader, 'set_epoch'):
            self.reader.set_epoch(count)

    def set_loader_cfg(
            self,
            num_workers: Optional[int] = None,
    ):
        # TFDS and WDS readers need # workers for correct # samples estimate before loader processes created
        if hasattr(self.reader, 'set_loader_cfg'):
            self.reader.set_loader_cfg(num_workers=num_workers)

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)
