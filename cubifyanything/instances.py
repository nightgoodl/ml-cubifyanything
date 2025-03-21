# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

# Based on D2's Instances.
import itertools
import warnings
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

# Provides basic compatibility with D2.
class Instances3D:
    """
    This class represents a list of instances in _the world_.
    """
    def __init__(self, image_size: Tuple[int, int] = (0, 0), **kwargs: Any):
        # image_size is here for Detectron2 compatibility.
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width (note: opposite of cubifycore).

        Here for D2 compatibility. You probably shouldn't be using this.
        """
        return self._image_size            

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances3D!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        with warnings.catch_warnings(record=True):
            data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Instances3D of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Instances3D":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Instances3D(image_size=self._image_size)
        # Copy fields that were explicitly added to this object (e.g., hidden fields)
        for name, value in self.__dict__.items():
            if (name not in ["_fields"]) and name.startswith("_"):
                setattr(ret, name, value.to(*args, **kwargs) if hasattr(value, "to") else value)
        
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)

        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances3D":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances3D` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances3D index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances3D(image_size=self.image_size)
        for name, value in self.__dict__.items():
            if (name not in ["_fields"]) and name.startswith("_"):
                setattr(ret, name, value)
        
        for k, v in self._fields.items():
            if isinstance(v, (torch.Tensor, np.ndarray)) or hasattr(v, "tensor"):
                # assume if has .tensor, then this is piped into __getitem__.
                # Make sure to match underlying types.
                if isinstance(v, np.ndarray) and isinstance(item, torch.Tensor):
                    ret.set(k, v[item.cpu().numpy()])
                else:
                    ret.set(k, v[item])
            elif hasattr(v, "__iter__"):
                # handle non-Tensor types like lists, etc.
                if isinstance(item, np.ndarray) and (item.dtype == np.bool_):
                    ret.set(k, [v_ for i_, v_ in enumerate(v) if item[i_]])                    
                elif isinstance(item, torch.BoolTensor) or (isinstance(item, torch.Tensor) and (item.dtype == torch.bool)):
                    ret.set(k, [v_ for i_, v_ in enumerate(v) if item[i_].item()])
                elif isinstance(item, torch.LongTensor) or (isinstance(item, torch.Tensor) and (item.dtype == torch.int64)):
                    # Can this be right?
                    ret.set(k, [v[i_.item()] for i_ in item])
                elif isinstance(item, slice):
                    ret.set(k, v[item])
                else:
                    raise ValueError("Expected Bool or Long Tensor")
            else:
                raise ValueError("Not supported!")
                
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()
        raise NotImplementedError("Empty Instances3D does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Instances3D` object is not iterable!")

    def split(self, split_size_or_sections):
        indexes = torch.arange(len(self))
        splits = torch.split(indexes, split_size_or_sections)

        return [self[split] for split in splits]

    def clone(self):
        import copy

        ret = Instances3D(image_size=self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "clone"):
                v = v.clone()
            elif isinstance(v, np.ndarray):
                v = np.copy(v)
            elif isinstance(v, (str, list, tuple)):
                v = copy.copy(v)
            elif hasattr(v, "tensor"):
                v = type(v)(v.tensor.clone())
            else:
                raise NotImplementedError

            ret.set(k, v)

        return ret

    @staticmethod
    def cat(instance_lists: List["Instances3D"]) -> "Instances3D":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances3D) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        ret = Instances3D(image_size=instance_lists[0]._image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def translate(self, translation):
        # in-place.
        for field_name, field in self._fields.items():
            if hasattr(field, "translate"):
                field.translate(translation)

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__
