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


class NewZeroOneValueTransform(ValueTransform):
    @staticmethod
    def normalize(values: th.Tensor) -> th.Tensor:
        # scale all values between one and zero
        # the values can contain negative infinity
        values = values.clone()
        min_finite_value = values[values.isfinite()].min()
        max_value = values.max()
        value_spread = max_value - min_finite_value
        if value_spread == 0:
            # set all finite values to 1, negative infinity to 0
            scaled_values = th.where(
                values != -th.inf, th.ones_like(values), th.zeros_like(values)
            )
        else:
            values[values == -th.inf] = min_finite_value
            scaled_values = (values - min_finite_value) / value_spread

        return scaled_values

class SoftMaxValueTransform(ValueTransform):
    @staticmethod
    def normalize(values: th.Tensor) -> th.Tensor:
        return th.nn.functional.softmax(values, dim=-1)



value_transform_dict = {
    "identity": IdentityValueTransform,
    "zero_one": ZeroOneValueTransform,
    "new_zero_one": NewZeroOneValueTransform,
    "softmax": SoftMaxValueTransform,
}
