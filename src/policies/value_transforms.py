from abc import ABC, abstractmethod
import torch as th

class ValueTransform(ABC):
    @staticmethod
    @abstractmethod
    def normalize(values: th.Tensor) -> th.Tensor:
        # This abstract static method defines the contract for normalization
        raise NotImplementedError

class IdentityValueTransform(ValueTransform):
    @staticmethod
    def normalize(values: th.Tensor) -> th.Tensor:
        return values

class ZeroOneValueTransform(ValueTransform):
    @staticmethod
    def normalize(values: th.Tensor) -> th.Tensor:
        values = values.clone()
        val = values[values.isfinite()]
        if val.numel() == 0:
            return th.zeros_like(values)
        max_val = val.max()
        min_val = val.min()

        if max_val == min_val:
            val = th.zeros_like(val)
        else:
            val = (val - min_val) / (max_val - min_val)
        values[values.isfinite()] = val
        return values


value_transform_dict = {
    "identity": IdentityValueTransform,
    "zero_one": ZeroOneValueTransform,
}
