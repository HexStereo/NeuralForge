"""Validation and conversion service for visual neural network graphs."""

from typing import Dict, Any, List, Tuple
from collections import defaultdict, deque
import json

from backend.app.schemas.architecture import ArchitectureConfig, ArchitectureValidationResult, LayerSummary
from neuralforge_core import Sequential, Dense, ReLU, Sigmoid, Tanh, Softmax, Flatten, Dropout, Module


class ArchitectureService:
    @staticmethod
    def validate_and_compile(config: ArchitectureConfig) -> Tuple[ArchitectureValidationResult, List[Dict[str, Any]]]:
        """Validate node graph, check dimensions, detect cycles, calculate parameters, and produce layer specs."""
        nodes = config.nodes
        edges = config.edges
        errors: List[str] = []
        layer_summaries: List[LayerSummary] = []
        compiled_layer_specs: List[Dict[str, Any]] = []

        if not nodes:
            return ArchitectureValidationResult(is_valid=False, errors=["Architecture has no nodes."]), []

        node_map = {n.id: n for n in nodes}
        adj = defaultdict(list)
        in_degree = defaultdict(int)

        for n in nodes:
            in_degree[n.id] = 0

        for edge in edges:
            if edge.source not in node_map:
                errors.append(f"Edge references non-existent source node: {edge.source}")
                continue
            if edge.target not in node_map:
                errors.append(f"Edge references non-existent target node: {edge.target}")
                continue
            adj[edge.source].append(edge.target)
            in_degree[edge.target] += 1

        if errors:
            return ArchitectureValidationResult(is_valid=False, errors=errors), []

        # Find start nodes (in_degree == 0)
        start_nodes = [nid for nid, deg in in_degree.items() if deg == 0]
        if not start_nodes:
            return ArchitectureValidationResult(is_valid=False, errors=["Graph contains cycles or has no input node."]), []

        # Check for multiple start nodes in sequential architecture
        input_nodes = [nid for nid in start_nodes if node_map[nid].type in ["input", "Input"]]
        if len(input_nodes) == 0:
            errors.append("Architecture must have an Input node.")
        elif len(input_nodes) > 1:
            errors.append("First version supports single-input feedforward architectures. Multiple input nodes detected.")

        if errors:
            return ArchitectureValidationResult(is_valid=False, errors=errors), []

        # Topological sort (Kahn's algorithm)
        queue = deque(start_nodes)
        topo_order: List[str] = []
        in_deg_copy = in_degree.copy()

        while queue:
            curr = queue.popleft()
            topo_order.append(curr)
            for neighbor in adj[curr]:
                in_deg_copy[neighbor] -= 1
                if in_deg_copy[neighbor] == 0:
                    queue.append(neighbor)

        if len(topo_order) != len(nodes):
            return ArchitectureValidationResult(is_valid=False, errors=["Cycle detected in architecture graph."]), []

        # Traverse sequential path and validate dimensions
        current_dim = config.input_dim
        total_parameters = 0
        layer_idx = 0

        for node_id in topo_order:
            node = node_map[node_id]
            ntype = node.type.lower()
            data = node.data or {}

            if ntype in ["input"]:
                # Dimension defined by dataset or node data
                current_dim = int(data.get("dim", current_dim))
                continue

            elif ntype in ["dense", "linear"]:
                in_f = int(data.get("in_features", current_dim))
                out_f = int(data.get("out_features", 0))
                bias = bool(data.get("bias", True))

                if out_f <= 0:
                    errors.append(f"Node '{node_id}' (Dense) must have out_features > 0.")
                    break

                if in_f != current_dim:
                    errors.append(
                        f"Dimension mismatch at node '{node_id}' (Dense): expected input dim {current_dim}, "
                        f"but node specifies {in_f}."
                    )
                    break

                params = (in_f * out_f) + (out_f if bias else 0)
                total_parameters += params

                layer_summaries.append(LayerSummary(
                    index=layer_idx,
                    name=f"dense_{layer_idx}",
                    type="Dense",
                    input_shape=[-1, in_f],
                    output_shape=[-1, out_f],
                    param_count=params
                ))
                compiled_layer_specs.append({
                    "type": "dense",
                    "in_features": in_f,
                    "out_features": out_f,
                    "bias": bias
                })
                current_dim = out_f
                layer_idx += 1

                # If node also specifies an activation
                act = data.get("activation")
                if act and act.lower() not in ["none", "linear"]:
                    act_type = act.lower()
                    compiled_layer_specs.append({"type": act_type})
                    layer_summaries.append(LayerSummary(
                        index=layer_idx,
                        name=f"{act_type}_{layer_idx}",
                        type=act.upper(),
                        input_shape=[-1, current_dim],
                        output_shape=[-1, current_dim],
                        param_count=0
                    ))
                    layer_idx += 1

            elif ntype in ["relu", "sigmoid", "tanh", "softmax"]:
                compiled_layer_specs.append({"type": ntype})
                layer_summaries.append(LayerSummary(
                    index=layer_idx,
                    name=f"{ntype}_{layer_idx}",
                    type=ntype.upper(),
                    input_shape=[-1, current_dim],
                    output_shape=[-1, current_dim],
                    param_count=0
                ))
                layer_idx += 1

            elif ntype in ["dropout"]:
                p = float(data.get("p", data.get("drop_rate", 0.5)))
                if not 0.0 <= p < 1.0:
                    errors.append(f"Dropout probability must be in [0, 1), got {p}")
                    break
                compiled_layer_specs.append({"type": "dropout", "p": p})
                layer_summaries.append(LayerSummary(
                    index=layer_idx,
                    name=f"dropout_{layer_idx}",
                    type="Dropout",
                    input_shape=[-1, current_dim],
                    output_shape=[-1, current_dim],
                    param_count=0
                ))
                layer_idx += 1

            elif ntype in ["flatten"]:
                compiled_layer_specs.append({"type": "flatten"})
                layer_summaries.append(LayerSummary(
                    index=layer_idx,
                    name=f"flatten_{layer_idx}",
                    type="Flatten",
                    input_shape=[-1, current_dim],
                    output_shape=[-1, current_dim],
                    param_count=0
                ))
                layer_idx += 1

            elif ntype in ["output"]:
                # Ensure current_dim matches target output dimension
                if config.output_dim and current_dim != config.output_dim:
                    errors.append(
                        f"Final network output dimension ({current_dim}) does not match expected "
                        f"target output dimension ({config.output_dim})."
                    )
                continue

            else:
                errors.append(f"Unsupported layer type: '{ntype}' in node '{node_id}'")
                break

        if errors:
            return ArchitectureValidationResult(is_valid=False, errors=errors), []

        return ArchitectureValidationResult(
            is_valid=True,
            errors=[],
            total_parameters=total_parameters,
            layer_summaries=layer_summaries
        ), compiled_layer_specs

    @staticmethod
    def build_core_model(layer_specs: List[Dict[str, Any]], seed: int = 42) -> Sequential:
        """Instantiate a real NeuralForge Core Sequential model from validated layer specs."""
        seq = Sequential()
        for i, spec in enumerate(layer_specs):
            ltype = spec["type"].lower()
            if ltype == "dense":
                seq.append(Dense(
                    in_features=spec["in_features"],
                    out_features=spec["out_features"],
                    bias=spec.get("bias", True),
                    seed=seed + i
                ))
            elif ltype == "relu":
                seq.append(ReLU())
            elif ltype == "sigmoid":
                seq.append(Sigmoid())
            elif ltype == "tanh":
                seq.append(Tanh())
            elif ltype == "softmax":
                seq.append(Softmax())
            elif ltype == "dropout":
                seq.append(Dropout(p=spec.get("p", 0.5), seed=seed + i))
            elif ltype == "flatten":
                seq.append(Flatten())
            else:
                raise ValueError(f"Unknown layer type in specs: {ltype}")
        return seq
