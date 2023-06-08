from typing import Callable
from functools import wraps
import torch.distributed as dist
import torch
import pickle



def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """

    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]

    rank = dist.get_rank()
    device = torch.device('cuda', rank)

    # serialized to a Tensor
    print(f'Dump to pickle')
    buffer = pickle.dumps(data)

    print(f'Gather to ByteStorage')
    storage = torch.ByteStorage.from_buffer(buffer)

    tensor = torch.ByteTensor(storage)

    print(f'Move to device')
    tensor = tensor.to(device)

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to(device)
    size_list = [torch.LongTensor([0]).to(device) for _ in range(world_size)]

    dist.barrier()

    print(f'Gather size list')
    dist.all_gather(size_list, local_size)


    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to(device))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to(device)
        tensor = torch.cat((tensor, padding), dim=0)

    dist.all_gather(tensor_list, tensor)


    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


class NotComputableError(RuntimeError):
    """
    Exception class to raise if Metric cannot be computed.
    """


def reinit__is_reduced(func: Callable) -> Callable:
    """Helper decorator for distributed configuration.
    See :doc:`metrics` on how to use it.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        self._is_reduced = False

    wrapper._decorated = True
    return wrapper


class EpochGather:
    def __init__(self, output_transform: Callable = lambda x: x):
        self._is_reduced = False
        self.reset()

    @reinit__is_reduced
    def reset(self) -> None:
        self._store_values = []

    @property
    def val(self) -> list:
        return self._store_values

    # def update(self, output: Sequence[torch.Tensor]) -> None:
    @reinit__is_reduced
    def update(self, val: list) -> None:
        self._store_values.extend(val)

    def gather(self) -> torch.Tensor:
        # print(f'WTF    {self._predictions} {self._targets} ///', self._predictions, self._targets)

        rank = dist.get_rank()
        device = torch.device('cuda', rank)

        # _store_values_tensor = torch.cat(self._store_values, dim=0).to(device) #.view(-1)

        _store_values_output = self._store_values
        ws = dist.get_world_size()

        dist.barrier()
        if ws > 1 and not self._is_reduced:

            print(f'Enter all_gather')
            _store_values_output = all_gather(_store_values_output)

        dist.barrier()
        self._is_reduced = True

        return _store_values_output