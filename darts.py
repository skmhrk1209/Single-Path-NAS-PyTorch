import torch
from torch import nn
from ops import *
from utils import *
from operator import *
import networkx as nx
import graphviz as gv


class DARTS(nn.Module):
    """Differentiable architecture search module.
    Based on the following papers.
    1. [DARTS: Differentiable Architecture Search](https://arxiv.org/pdf/1806.09055.pdf)
    2. ...
    """

    def __init__(self, operations, num_nodes, num_input_nodes, num_cells, reduction_cells,
                 num_predecessors, num_channels, num_classes, drop_prob_fn):
        """Build DARTS with the given operations.
        Args:
            operations (dict): Dict with name as keys and nn.Module initializer
                that takes in_channels, out_channels, stride as arguments as values.
            num_nodes (int): Number of nodes in each cell.
            num_input_nodes (int): Number of input nodes in each cell.
            num_cells (int): Number of cells in the network.
            reduction_cells (list): List of cell index that performs spatial reduction.
            num_top_operations (int): Number of top strongest operations retained in discrete architecture.
            num_channels (int): Number of channels of the first cell.
            num_classes (int): Number of classes for classification.
        """
        super().__init__()

        self.operations = operations
        self.num_nodes = num_nodes
        self.num_input_nodes = num_input_nodes
        self.num_cells = num_cells
        self.reduction_cells = reduction_cells
        self.num_predecessors = num_predecessors
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.scheduled_drop_path = ScheduledDropPath(drop_prob_fn)

        self.build_dag()
        self.build_architecture()
        self.build_network()

    def build_dag(self):
        """Build Directed Acyclic Graph that represents each cell.
        """
        self.dag = Dict()
        self.dag.normal = nx.DiGraph()
        for n in range(self.num_input_nodes, self.num_nodes):
            for m in range(self.num_nodes):
                if m < n:
                    self.dag.normal.add_edge(m, n)
        self.dag.reduction = nx.DiGraph()
        for n in range(self.num_input_nodes, self.num_nodes):
            for m in range(self.num_nodes):
                if m < n:
                    self.dag.reduction.add_edge(m, n)

    def build_discrete_dag(self):
        """Build Directed Acyclic Graph that represents each cell.
        """
        for n in self.dag.normal.nodes():
            predecessors = (
                (torch.max(nn.functional.softmax(self.architecture.normal[str((m, n))], dim=0)), m)
                for m in self.dag.normal.predecessors(n)
            )
            predecessors = sorted(predecessors, key=itemgetter(0))
            for weight, m in predecessors[:-self.num_predecessors]:
                self.dag.normal.remove_edge(m, n)

        for n in self.dag.reduction.nodes():
            predecessors = (
                (torch.max(nn.functional.softmax(self.architecture.reduction[str((m, n))], dim=0)), m)
                for m in self.dag.reduction.predecessors(n)
            )
            predecessors = sorted(predecessors, key=itemgetter(0))
            for weight, m in predecessors[:-self.num_predecessors]:
                self.dag.reduction.remove_edge(m, n)

    def build_architecture(self):
        """Build parameters that represent the cell architectures (normal and reduction).
        """
        self.architecture = nn.ParameterDict()
        self.architecture.normal = nn.ParameterDict({
            str((m, n)): nn.Parameter(torch.zeros(len(self.operations)))
            for m, n in self.dag.normal.edges()
        })
        self.architecture.reduction = nn.ParameterDict({
            str((m, n)): nn.Parameter(torch.zeros(len(self.operations)))
            for m, n in self.dag.reduction.edges()
        })

    def build_network(self):
        """Build modules that represent the whole network.
        """
        self.network = nn.ParameterDict()

        # NOTE: Why multiplier is 3?
        num_channels = self.num_channels
        out_channels = num_channels * 3

        self.network.conv = Conv2d(
            in_channels=3,
            out_channels=out_channels,
            stride=1,
            kernel_size=3,
            padding=1,
            affine=True,
            preactivation=False
        )

        out_channels = [out_channels] * self.num_input_nodes
        self.network.cells = nn.ModuleList()

        for i in range(self.num_cells):

            reduction = i in self.reduction_cells
            num_channels = num_channels << 1 if reduction else num_channels

            dag = self.dag.reduction if reduction else self.dag.normal
            architecture = self.architecture.reduction if reduction else self.architecture.normal

            cell = nn.ModuleDict({
                # NOTE: Should be factorized reduce?
                **{
                    str((n - self.num_input_nodes, n)): Conv2d(
                        in_channels=out_channels[n - self.num_input_nodes],
                        out_channels=num_channels,
                        stride=1 << len([j for j in self.reduction_cells if k < j < i]),
                        kernel_size=1,
                        padding=0,
                        affine=False
                    ) for n, k in zip(range(self.num_input_nodes), range(i - self.num_input_nodes, i))
                },
                **{
                    str((m, n)): nn.ModuleList([
                        operation(
                            in_channels=num_channels,
                            out_channels=num_channels,
                            stride=2 if reduction and m in range(self.num_input_nodes) else 1,
                            affine=False
                        ) for operation in self.operations.values()
                    ]) for m, n in dag.edges()
                }
            })

            out_channels.append(num_channels * (self.num_nodes - self.num_input_nodes))
            self.network.cells.append(cell)

        self.network.global_avg_pool2d = nn.AdaptiveAvgPool2d(1)
        self.network.linear = nn.Linear(out_channels[-1], self.num_classes, bias=True)

    def build_discrete_network(self):
        """Build modules that represent the whole network.
        """
        self.network = nn.ParameterDict()

        # NOTE: Why multiplier is 3?
        num_channels = self.num_channels
        out_channels = num_channels * 3

        self.network.conv = Conv2d(
            in_channels=3,
            out_channels=out_channels,
            stride=1,
            kernel_size=3,
            padding=1,
            affine=True,
            preactivation=False
        )

        out_channels = [out_channels] * self.num_input_nodes
        self.network.cells = nn.ModuleList()

        for i in range(self.num_cells):

            reduction = i in self.reduction_cells
            num_channels = num_channels << 1 if reduction else num_channels

            dag = self.dag.reduction if reduction else self.dag.normal
            architecture = self.architecture.reduction if reduction else self.architecture.normal

            cell = nn.ModuleDict({
                # NOTE: Should be factorized reduce?
                **{
                    str((n - self.num_input_nodes, n)): Conv2d(
                        in_channels=out_channels[n - self.num_input_nodes],
                        out_channels=num_channels,
                        stride=1 << len([j for j in self.reduction_cells if k < j < i]),
                        kernel_size=1,
                        padding=0,
                        affine=True
                    ) for n, k in zip(range(self.num_input_nodes), range(i - self.num_input_nodes, i))
                },
                **{
                    str((m, n)): max(((weight, operation) for weight, (name, operation) in zip(
                        nn.functional.softmax(architecture[str((m, n))], dim=0),
                        self.operations.items()
                    ) if 'zero' not in name), key=itemgetter(0))[1](
                        in_channels=num_channels,
                        out_channels=num_channels,
                        stride=2 if reduction and m in range(self.num_input_nodes) else 1,
                        affine=True
                    ) for m, n in dag.edges()
                }
            })

            out_channels.append(num_channels * (self.num_nodes - self.num_input_nodes))
            self.network.cells.append(cell)

        self.network.global_avg_pool2d = nn.AdaptiveAvgPool2d(1)
        self.network.linear = nn.Linear(out_channels[-1], self.num_classes, bias=True)

    def forward_cell(self, cell, reduction, cell_outputs, node_outputs, n):
        """forward in the given cell.
        Args:
            cell (dict): A dict with edges as keys and operations as values.
            reduction (bool): Whether the cell performs spatial reduction.
            node_outputs (dict): A dict with node as keys and its outputs as values.
                This is to avoid duplicate calculation in recursion.
            n (int): The output node in the cell.
        """
        dag = self.dag.reduction if reduction else self.dag.normal
        architecture = self.architecture.reduction if reduction else self.architecture.normal

        if n not in node_outputs:
            if n in range(self.num_input_nodes):
                node_outputs[n] = cell[str((n - self.num_input_nodes, n))](cell_outputs[n - self.num_input_nodes])
            else:
                node_outputs[n] = sum(self.scheduled_drop_path(
                    sum(weight * operation(self.forward_cell(cell, reduction, cell_outputs, node_outputs, m))
                        for weight, operation in zip(nn.functional.softmax(architecture[str((m, n))], dim=0), cell[str((m, n))]))
                    if isinstance(cell[str((m, n))], nn.ModuleList) else
                    cell[str((m, n))](self.forward_cell(cell, reduction, cell_outputs, node_outputs, m))
                ) for m in dag.predecessors(n))
        return node_outputs[n]

    def forward(self, input):
        output = self.network.conv(input)
        cell_outputs = [output] * self.num_input_nodes
        for i, cell in enumerate(self.network.cells):
            node_outputs = {}
            cell_outputs.append(torch.cat([
                self.forward_cell(cell, i in self.reduction_cells, cell_outputs, node_outputs, n)
                for n in range(self.num_input_nodes, self.num_nodes)
            ], dim=1))
        output = cell_outputs[-1]
        output = self.network.global_avg_pool2d(output).squeeze()
        output = self.network.linear(output)
        return output

    def super_kernel_activation_maps(self):
        return {
            f'{i}_{edge}': torch.norm(submodule.super_weight, dim=(0, 1))
            for i, cell in enumerate(self.network.cells)
            for edge, module in cell.items()
            for submodule in module.modules()
            if isinstance(submodule, SuperConv2d)
        }

    def render_discrete_architecture(self, reduction, name, directory):
        """Render the given architecture.
        Args:
            architecture (dict): A dict with edges as keys and parameters as values.
            name (str): Name of the given architecture for saving.
            directory (str): Directory for saving.
        """
        dag = self.dag.reduction if reduction else self.dag.normal
        architecture = self.architecture.reduction if reduction else self.architecture.normal

        discrete_dag = gv.Digraph(name)
        for n in dag.nodes():
            operations = sorted(((*max(((weight, operation) for weight, operation in zip(
                nn.functional.softmax(architecture[str((m, n))], dim=0),
                self.operations.keys()
            ) if 'zero' not in operation), key=itemgetter(0)), m) for m in dag.predecessors(n)), key=itemgetter(0))
            for weight, operation, m in operations[:-self.num_predecessors]:
                discrete_dag.edge(str(m), str(n), label='', color='white')
            for weight, operation, m in operations[-self.num_predecessors:]:
                discrete_dag.edge(str(m), str(n), label=operation, color='black')
        return discrete_dag.render(directory=directory, format='png')

    def set_epoch(self, epoch):
        self.scheduled_drop_path.set_epoch(epoch)
