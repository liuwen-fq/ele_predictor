"""Fragment-level encoding with legacy and sidecar fusion modes."""

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter, softmax

from formula_design.data import Data


class FragmentBlock(nn.Module):
    """Aggregate atom/bond states into fragments and connect fragment context."""

    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int = 64,
                 connection_feature_dim: int = 9,
                 num_message_layers: int = 1,
                 gate_init: float = -4.0,
                 neutral_only: bool = True,
                 mode: str = "legacy",
                 sidecar_gate_init: float = 0.0,
                 sidecar_gate_scale: float = 0.1):
        super().__init__()
        if num_message_layers < 0:
            raise ValueError("num_message_layers must be non-negative")
        if mode not in {"legacy", "sidecar"}:
            raise ValueError(f"unsupported fragment mode: {mode}")
        if sidecar_gate_scale <= 0:
            raise ValueError("sidecar_gate_scale must be positive")

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.gate_init = float(gate_init)
        self.neutral_only = neutral_only
        self.mode = mode
        self.sidecar_gate_init = float(sidecar_gate_init)
        self.sidecar_gate_scale = float(sidecar_gate_scale)

        self.fragment_input = nn.Sequential(
            nn.Linear(node_dim + edge_dim + 1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.connection_encoder = nn.Sequential(
            nn.Linear(connection_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(num_message_layers)
        ])
        self.update_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            ) for _ in range(num_message_layers)
        ])
        self.fragment_score = nn.Linear(hidden_dim, 1)
        if self.mode == "legacy":
            self.node_projection = nn.Linear(hidden_dim, node_dim)
            self.edge_projection = nn.Linear(hidden_dim, edge_dim)
            self.node_gate_logit = nn.Parameter(torch.tensor(self.gate_init))
            self.edge_gate_logit = nn.Parameter(torch.tensor(self.gate_init))
        else:
            molecule_dim = node_dim + edge_dim
            self.molecule_projection = nn.Sequential(
                nn.Linear(hidden_dim, molecule_dim),
                nn.GELU(),
                nn.LayerNorm(molecule_dim),
            )
            self.molecule_gate_logit = nn.Parameter(
                torch.tensor(self.sidecar_gate_init))

        self.last_fragment_attention = None
        self.last_fragment_assignment = None
        self.last_fragment_embedding = None
        self.last_fragment_batch = None
        self.last_fragment_active = None
        self.last_molecule_embedding = None

    def reset_parameters(self):
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        with torch.no_grad():
            if self.mode == "legacy":
                self.node_gate_logit.fill_(self.gate_init)
                self.edge_gate_logit.fill_(self.gate_init)
            else:
                self.molecule_gate_logit.fill_(self.sidecar_gate_init)
        self.last_fragment_attention = None
        self.last_fragment_assignment = None
        self.last_fragment_embedding = None
        self.last_fragment_batch = None
        self.last_fragment_active = None
        self.last_molecule_embedding = None

    @staticmethod
    def _batch_index(counts: Tensor, device: torch.device) -> Tensor:
        return torch.arange(counts.shape[0], dtype=torch.long,
                            device=device).repeat_interleave(counts.long())

    @staticmethod
    def _validate_data(data: Data):
        required = (
            "inc_fragment_atom",
            "fragment_attachment",
            "inc_fragment_connection",
            "fragment_connection_features",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(
                "fragment encoder requires reprocessed data; missing fields: "
                + ", ".join(missing))
        if data.counts.shape[1] < 3:
            raise ValueError(
                "fragment encoder requires fragment counts in data.counts")

    def _fragment_embeddings(self, data: Data, node_h: Tensor,
                             edge_h: Tensor) -> tuple[Tensor, Tensor]:
        fragment_index = data.inc_fragment_atom.long()
        num_fragments = int(data.counts[:, 2].sum().item())

        node_fragment_h = scatter(
            node_h,
            fragment_index,
            dim=0,
            dim_size=num_fragments,
            reduce="mean",
        )

        edge_endpoints = data.inc_node_edge.long()
        if edge_h.shape[0] > 0:
            edge_dst_fragment = fragment_index[edge_endpoints[:, 0]]
            edge_src_fragment = fragment_index[edge_endpoints[:, 1]]
            internal_mask = edge_dst_fragment == edge_src_fragment
            internal_edge_h = scatter(
                edge_h[internal_mask],
                edge_dst_fragment[internal_mask],
                dim=0,
                dim_size=num_fragments,
                reduce="mean",
            )
        else:
            internal_edge_h = edge_h.new_zeros((num_fragments, self.edge_dim))

        attachment_fraction = scatter(
            data.fragment_attachment.to(node_h.dtype),
            fragment_index,
            dim=0,
            dim_size=num_fragments,
            reduce="mean",
        )
        fragment_h = self.fragment_input(
            torch.cat(
                [node_fragment_h, internal_edge_h, attachment_fraction],
                dim=-1,
            ))
        return fragment_h, fragment_index

    def _connection_message_passing(self, data: Data,
                                    fragment_h: Tensor) -> Tensor:
        connection_index = data.inc_fragment_connection.long()
        if connection_index.shape[0] == 0 or not self.message_mlps:
            return fragment_h

        connection_h = self.connection_encoder(
            data.fragment_connection_features.to(fragment_h.dtype))
        dst = connection_index[:, 0]
        src = connection_index[:, 1]
        for message_mlp, update_mlp in zip(self.message_mlps,
                                           self.update_mlps):
            messages = message_mlp(
                torch.cat([fragment_h[src], connection_h], dim=-1))
            aggregated = scatter(
                messages,
                dst,
                dim=0,
                dim_size=fragment_h.shape[0],
                reduce="sum",
            )
            fragment_h = fragment_h + update_mlp(
                torch.cat([fragment_h, aggregated], dim=-1))
        return fragment_h

    def _pool_molecules(self, data: Data,
                        fragment_h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        fragment_counts = data.counts[:, 2]
        fragment_batch = self._batch_index(fragment_counts, fragment_h.device)
        logits = self.fragment_score(fragment_h).squeeze(-1)
        attention = softmax(logits, fragment_batch)
        molecule_h = scatter(
            fragment_h * attention.unsqueeze(-1),
            fragment_batch,
            dim=0,
            dim_size=fragment_counts.shape[0],
            reduce="sum",
        )
        return molecule_h, attention, fragment_batch

    def forward(self, data: Data, node_h: Tensor,
                edge_h: Tensor) -> tuple[Tensor, Tensor]:
        self._validate_data(data)
        fragment_h, fragment_index = self._fragment_embeddings(
            data, node_h, edge_h)
        fragment_h = self._connection_message_passing(data, fragment_h)
        molecule_h, attention, fragment_batch = self._pool_molecules(
            data, fragment_h)

        node_batch = self._batch_index(data.counts[:, 0], node_h.device)
        neutral_mask = node_h.new_ones(data.counts.shape[0])
        if self.neutral_only:
            molecule_charge = scatter(
                data.node_features[:, 2].to(node_h.dtype),
                node_batch,
                dim=0,
                dim_size=data.counts.shape[0],
                reduce="sum",
            )
            neutral_mask = (molecule_charge == 0).to(node_h.dtype)
            molecule_h = molecule_h * neutral_mask.unsqueeze(-1)

        self.last_fragment_attention = attention.detach()
        self.last_fragment_assignment = fragment_index.detach()
        self.last_fragment_embedding = fragment_h
        self.last_fragment_batch = fragment_batch.detach()
        self.last_fragment_active = neutral_mask[fragment_batch].detach()
        self.last_molecule_embedding = molecule_h

        if self.mode == "sidecar":
            return node_h, edge_h

        edge_batch = self._batch_index(data.counts[:, 1], edge_h.device)
        node_delta = self.node_projection(molecule_h)[node_batch]
        edge_delta = self.edge_projection(molecule_h)[edge_batch]
        if self.neutral_only:
            node_delta = node_delta * neutral_mask[node_batch].unsqueeze(-1)
            edge_delta = edge_delta * neutral_mask[edge_batch].unsqueeze(-1)
        node_gate = torch.sigmoid(self.node_gate_logit)
        edge_gate = torch.sigmoid(self.edge_gate_logit)

        return node_h + node_gate * node_delta, edge_h + edge_gate * edge_delta

    def fuse_molecule_embedding(self, base_embedding: Tensor) -> Tensor:
        """Fuse the sidecar molecule vector without changing graph states."""

        if self.mode != "sidecar":
            return base_embedding
        if self.last_molecule_embedding is None:
            raise RuntimeError("fragment embedding requested before encoding")
        delta = self.molecule_projection(self.last_molecule_embedding)
        gate = self.sidecar_gate_scale * torch.tanh(self.molecule_gate_logit)
        return base_embedding + gate * delta

    def get_molecule_embedding(self, detach: bool = False) -> Tensor:
        if self.last_molecule_embedding is None:
            raise RuntimeError("fragment embedding requested before encoding")
        if detach:
            return self.last_molecule_embedding.detach()
        return self.last_molecule_embedding

    def get_fragment_embeddings(
            self, detach: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        """Return fragment states and their molecule assignment.

        ``active`` is zero for charged molecules when ``neutral_only`` is
        enabled.  Formula-level interpretable heads use these tensors without
        altering the Bamboo atom/bond representation.
        """

        if self.last_fragment_embedding is None:
            raise RuntimeError("fragment embedding requested before encoding")
        values = (
            self.last_fragment_embedding,
            self.last_fragment_batch,
            self.last_fragment_active,
        )
        if detach:
            return tuple(value.detach() for value in values)
        return values

    def sidecar_gate(self) -> Tensor:
        if self.mode != "sidecar":
            raise RuntimeError("sidecar gate is unavailable in legacy mode")
        return self.sidecar_gate_scale * torch.tanh(self.molecule_gate_logit)
