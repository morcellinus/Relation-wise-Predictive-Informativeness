"""Compute operational RPI on a small author-paper graph."""

import torch
from torch_geometric.data import HeteroData

from rpi import compute_rpi


def main() -> None:
    graph = HeteroData()
    graph["author"].num_nodes = 4
    graph["author"].y = torch.tensor([0, 1, -1, -1])
    graph["paper"].num_nodes = 2

    writes = torch.tensor([[0, 1, 2, 3], [0, 1, 0, 1]])
    graph["author", "writes", "paper"].edge_index = writes
    graph["paper", "written_by", "author"].edge_index = writes.flip(0)
    train_mask = torch.tensor([True, True, False, False])

    result = compute_rpi(graph, target_type="author", train_mask=train_mask)
    for row in result.relations:
        print(f"{row['relation']}: RPI={row['RPI']:.3f} ({row['num_edges']} edges)")
    if result.target is not None:
        print(f"Target {result.target['node_type']}: RPI={result.target['RPI_type']:.3f}")


if __name__ == "__main__":
    main()
