from typing import (
    Dict,
    Any,
    List
)

import h5py
from matplotlib import pyplot


def save_data(filename: str, data: Dict[str, Any]):
    with h5py.File(name=filename, mode="w") as f:
        for k, v in data.items():
            if isinstance(v, dict):
                subgroup = f.create_group(k)
                for subk, subv in data[k].items():
                    subgroup.create_dataset(name=subk, data=subv)
            else:
                f.create_dataset(name=k, data=v)

def plot_costs(costs: List[float]):
    fig, ax = pyplot.subplots()
    ax.plot(range(len(costs)), costs, 'b+')
    ax.set(xlabel='epoch', ylabel='cost', title='cost per epoch')
    pyplot.savefig("costs.png", dpi=400)