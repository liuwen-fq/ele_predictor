"""Chemistry-constrained molecular fragmentation.

Fragments are an additional view of the original molecule. Cross-fragment
bonds are retained as connection edges, so the original graph topology is not
lost.
"""

from dataclasses import dataclass
from typing import Iterable

from rdkit import Chem
from rdkit.Chem import BRICS

FRAGMENT_SCHEMA_VERSION = 1
FRAGMENT_RULE_VERSION = 1

# Keep the core of common solvent functional groups intact. BRICS may still cut
# bonds from these cores to substituents.
_PROTECTED_SMARTS = (
    "[CX3](=[OX1])[OX2]",  # ester
    "[OX2][CX3](=[OX1])[OX2]",  # carbonate
    "[NX3][CX3](=[OX1])",  # amide
    "[SX4](=[OX1])(=[OX1])",  # sulfone/sulfonyl core
    "[CX2]#[NX1]",  # nitrile
)
_PROTECTED_PATTERNS = tuple(pattern
                            for pattern in (Chem.MolFromSmarts(smarts)
                                            for smarts in _PROTECTED_SMARTS)
                            if pattern is not None)


@dataclass(frozen=True)
class FragmentGraph:
    """Fragment assignment and directed fragment-connection graph."""

    atom_to_fragment: tuple[int, ...]
    attachment_atoms: tuple[bool, ...]
    connection_index: tuple[tuple[int, int], ...]
    connection_features: tuple[tuple[int, ...], ...]
    cut_bond_indices: tuple[int, ...]
    num_fragments: int
    schema_version: int = FRAGMENT_SCHEMA_VERSION
    rule_version: int = FRAGMENT_RULE_VERSION


def _protected_bond_indices(mol: Chem.Mol) -> set[int]:
    protected = {bond.GetIdx() for bond in mol.GetBonds() if bond.IsInRing()}
    for pattern in _PROTECTED_PATTERNS:
        for match in mol.GetSubstructMatches(pattern):
            for pattern_bond in pattern.GetBonds():
                begin_idx = match[pattern_bond.GetBeginAtomIdx()]
                end_idx = match[pattern_bond.GetEndAtomIdx()]
                bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)
                if bond is not None:
                    protected.add(bond.GetIdx())
    return protected


def _brics_candidate_bonds(mol: Chem.Mol) -> set[int]:
    candidates = set()
    for (begin_idx, end_idx), _ in BRICS.FindBRICSBonds(mol):
        bond = mol.GetBondBetweenAtoms(int(begin_idx), int(end_idx))
        if bond is not None:
            candidates.add(bond.GetIdx())
    return candidates


def _components(mol: Chem.Mol,
                cut_bond_indices: Iterable[int]) -> list[list[int]]:
    cut_bond_indices = set(cut_bond_indices)
    adjacency = [[] for _ in range(mol.GetNumAtoms())]
    for bond in mol.GetBonds():
        if bond.GetIdx() in cut_bond_indices:
            continue
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        adjacency[begin_idx].append(end_idx)
        adjacency[end_idx].append(begin_idx)

    components = []
    visited = set()
    for start_idx in range(mol.GetNumAtoms()):
        if start_idx in visited:
            continue
        stack = [start_idx]
        visited.add(start_idx)
        component = []
        while stack:
            atom_idx = stack.pop()
            component.append(atom_idx)
            for neighbor_idx in adjacency[atom_idx]:
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    stack.append(neighbor_idx)
        components.append(sorted(component))
    return sorted(components, key=lambda atoms: atoms[0])


def _heavy_atom_count(mol: Chem.Mol, atom_indices: Iterable[int]) -> int:
    return sum(
        mol.GetAtomWithIdx(idx).GetAtomicNum() > 1 for idx in atom_indices)


def _is_meaningful_single_carbon_substituent(
        mol: Chem.Mol, atom_indices: Iterable[int]) -> bool:
    """Keep terminal methyl/methylene substituents as recognizable fragments."""

    atom_indices = list(atom_indices)
    heavy_atoms = [
        mol.GetAtomWithIdx(idx) for idx in atom_indices
        if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1
    ]
    if len(heavy_atoms) != 1 or heavy_atoms[0].GetAtomicNum() != 6:
        return False
    hydrogen_count = sum(
        mol.GetAtomWithIdx(idx).GetAtomicNum() == 1 for idx in atom_indices)
    return hydrogen_count >= 2


def _merge_small_components(mol: Chem.Mol, cut_bond_indices: set[int],
                            min_fragment_heavy_atoms: int) -> set[int]:
    """Restore cut bonds until every fragment has enough heavy atoms."""

    cuts = set(cut_bond_indices)
    while cuts:
        components = _components(mol, cuts)
        small_components = [
            set(component) for component in components
            if _heavy_atom_count(mol, component) < min_fragment_heavy_atoms
            and not _is_meaningful_single_carbon_substituent(mol, component)
        ]
        if not small_components:
            break

        restored = False
        for component in small_components:
            boundary_bonds = []
            for bond_idx in sorted(cuts):
                bond = mol.GetBondWithIdx(bond_idx)
                begin_in = bond.GetBeginAtomIdx() in component
                end_in = bond.GetEndAtomIdx() in component
                if begin_in != end_in:
                    boundary_bonds.append(bond_idx)
            if boundary_bonds:
                cuts.remove(boundary_bonds[0])
                restored = True
                break
        if not restored:
            break
    return cuts


def _connection_feature(bond: Chem.Bond, begin_atom: Chem.Atom,
                        end_atom: Chem.Atom) -> tuple[int, ...]:
    return (
        int(bond.IsInRing()),
        int(round(float(bond.GetBondTypeAsDouble()) * 2)),
        int(bond.GetIsAromatic()),
        begin_atom.GetAtomicNum(),
        end_atom.GetAtomicNum(),
        begin_atom.GetDegree(),
        end_atom.GetDegree(),
        begin_atom.GetFormalCharge(),
        end_atom.GetFormalCharge(),
    )


def build_fragment_graph(mol: Chem.Mol,
                         min_fragment_heavy_atoms: int = 2) -> FragmentGraph:
    """Build fragments and preserve every cross-fragment chemical bond.

    Atom indices are kept identical to ``mol``. Hydrogens naturally remain in
    the same connected component as their neighboring heavy atom.
    """

    if mol is None:
        raise ValueError("mol must not be None")
    if min_fragment_heavy_atoms < 1:
        raise ValueError("min_fragment_heavy_atoms must be positive")

    protected = _protected_bond_indices(mol)
    candidate_cuts = {
        bond_idx
        for bond_idx in _brics_candidate_bonds(mol)
        if bond_idx not in protected
    }
    candidate_cuts = _merge_small_components(mol, candidate_cuts,
                                             min_fragment_heavy_atoms)

    components = _components(mol, candidate_cuts)
    atom_to_fragment = [-1] * mol.GetNumAtoms()
    for fragment_idx, component in enumerate(components):
        for atom_idx in component:
            atom_to_fragment[atom_idx] = fragment_idx

    attachment_atoms = [False] * mol.GetNumAtoms()
    connection_index = []
    connection_features = []
    for bond_idx in sorted(candidate_cuts):
        bond = mol.GetBondWithIdx(bond_idx)
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        begin_fragment = atom_to_fragment[begin_idx]
        end_fragment = atom_to_fragment[end_idx]
        attachment_atoms[begin_idx] = True
        attachment_atoms[end_idx] = True

        # Store both directions to match the molecular graph convention.
        connection_index.append((end_fragment, begin_fragment))
        connection_features.append(
            _connection_feature(bond, mol.GetAtomWithIdx(begin_idx),
                                mol.GetAtomWithIdx(end_idx)))
        connection_index.append((begin_fragment, end_fragment))
        connection_features.append(
            _connection_feature(bond, mol.GetAtomWithIdx(end_idx),
                                mol.GetAtomWithIdx(begin_idx)))

    fragment_graph = FragmentGraph(
        atom_to_fragment=tuple(atom_to_fragment),
        attachment_atoms=tuple(attachment_atoms),
        connection_index=tuple(connection_index),
        connection_features=tuple(connection_features),
        cut_bond_indices=tuple(sorted(candidate_cuts)),
        num_fragments=len(components),
    )
    validate_fragment_graph(mol, fragment_graph)
    return fragment_graph


def build_fragment_metadata(mol: Chem.Mol,
                            fragment_graph: FragmentGraph) -> dict:
    """Return a JSON-serializable description of a fragment graph.

    Atom indices are zero-based RDKit indices of ``mol``. Connections are
    represented once per cut bond, unlike ``connection_index`` which stores
    both directed message-passing edges.
    """

    components = [[] for _ in range(fragment_graph.num_fragments)]
    for atom_idx, fragment_idx in enumerate(fragment_graph.atom_to_fragment):
        components[fragment_idx].append(atom_idx)

    fragments = []
    for fragment_idx, atom_indices in enumerate(components):
        fragments.append({
            "fragment_id":
            fragment_idx,
            "smiles":
            Chem.MolFragmentToSmiles(
                mol,
                atomsToUse=atom_indices,
                canonical=True,
                isomericSmiles=True,
            ),
            "atom_indices":
            atom_indices,
            "atom_map_numbers": [
                mol.GetAtomWithIdx(atom_idx).GetAtomMapNum()
                for atom_idx in atom_indices
            ],
            "attachment_atom_indices": [
                atom_idx for atom_idx in atom_indices
                if fragment_graph.attachment_atoms[atom_idx]
            ],
        })

    connections = []
    for bond_idx in fragment_graph.cut_bond_indices:
        bond = mol.GetBondWithIdx(bond_idx)
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        connections.append({
            "bond_index":
            bond_idx,
            "fragment_ids": [
                fragment_graph.atom_to_fragment[begin_idx],
                fragment_graph.atom_to_fragment[end_idx],
            ],
            "atom_indices": [begin_idx, end_idx],
            "atom_map_numbers": [
                mol.GetAtomWithIdx(begin_idx).GetAtomMapNum(),
                mol.GetAtomWithIdx(end_idx).GetAtomMapNum(),
            ],
            "bond_type":
            str(bond.GetBondType()),
            "is_aromatic":
            bool(bond.GetIsAromatic()),
        })

    return {
        "fragment_schema_version": fragment_graph.schema_version,
        "fragment_rule_version": fragment_graph.rule_version,
        "formal_charge":
        sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
        "fragments": fragments,
        "connections": connections,
    }


def attach_fragment_attention(metadata_by_formula: list[list[dict]],
                              attention_values: list[float],
                              neutral_only: bool) -> list[list[dict]]:
    """Attach per-fragment pooling weights to serialized molecular metadata."""

    offset = 0
    output = []
    for formula_metadata in metadata_by_formula:
        formula_output = []
        for molecule_metadata in formula_metadata:
            fragments = molecule_metadata["fragments"]
            next_offset = offset + len(fragments)
            if next_offset > len(attention_values):
                raise ValueError("fragment attention is shorter than metadata")

            annotated_fragments = []
            for fragment, attention in zip(
                    fragments, attention_values[offset:next_offset]):
                annotated_fragment = dict(fragment)
                annotated_fragment["pooling_attention"] = float(attention)
                annotated_fragments.append(annotated_fragment)

            formula_output.append({
                "molecule_name":
                molecule_metadata["molecule_name"],
                "mapped_smiles":
                molecule_metadata["mapped_smiles"],
                "formal_charge":
                molecule_metadata["formal_charge"],
                "fragment_encoder_applied":
                (not neutral_only or molecule_metadata["formal_charge"] == 0),
                "fragments":
                annotated_fragments,
                "connections":
                molecule_metadata["connections"],
            })
            offset = next_offset
        output.append(formula_output)

    if offset != len(attention_values):
        raise ValueError("fragment attention is longer than metadata")
    return output


def validate_fragment_graph(mol: Chem.Mol,
                            fragment_graph: FragmentGraph) -> None:
    """Validate that fragments and connection edges preserve molecular topology."""

    if len(fragment_graph.atom_to_fragment) != mol.GetNumAtoms():
        raise ValueError("fragment atom assignment does not match molecule")
    if any(idx < 0 or idx >= fragment_graph.num_fragments
           for idx in fragment_graph.atom_to_fragment):
        raise ValueError("invalid fragment index in atom assignment")
    if len(fragment_graph.connection_index) != 2 * len(
            fragment_graph.cut_bond_indices):
        raise ValueError("every cut bond must create two directed connections")
    if len(fragment_graph.connection_features) != len(
            fragment_graph.connection_index):
        raise ValueError(
            "connection features and indices have different sizes")

    cut_bonds = set(fragment_graph.cut_bond_indices)
    for bond in mol.GetBonds():
        begin_fragment = fragment_graph.atom_to_fragment[
            bond.GetBeginAtomIdx()]
        end_fragment = fragment_graph.atom_to_fragment[bond.GetEndAtomIdx()]
        is_cross_fragment = begin_fragment != end_fragment
        if is_cross_fragment != (bond.GetIdx() in cut_bonds):
            raise ValueError(
                f"bond {bond.GetIdx()} is not represented consistently")
