import unittest

try:
    from rdkit import Chem
except ModuleNotFoundError:
    Chem = None


@unittest.skipIf(Chem is None, "rdkit is not installed")
class FragmentationTest(unittest.TestCase):

    @staticmethod
    def _build(smiles):
        from formula_design.fragment import build_fragment_graph

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        return mol, build_fragment_graph(mol)

    def test_fragmentation_is_deterministic(self):
        _, first = self._build("CCOC(=O)OCC")
        _, second = self._build("CCOC(=O)OCC")
        self.assertEqual(first, second)

    def test_ring_bonds_are_not_cut(self):
        mol, fragment_graph = self._build("O=C1OCCO1")
        cut_bonds = set(fragment_graph.cut_bond_indices)
        for bond in mol.GetBonds():
            if bond.IsInRing():
                self.assertNotIn(bond.GetIdx(), cut_bonds)

    def test_cross_fragment_bonds_are_reconstructable(self):
        from formula_design.fragment import validate_fragment_graph

        mol, fragment_graph = self._build("CCOC(=O)OCC")
        validate_fragment_graph(mol, fragment_graph)
        self.assertEqual(
            len(fragment_graph.connection_index),
            2 * len(fragment_graph.cut_bond_indices),
        )

    def test_small_molecule_falls_back_to_one_fragment(self):
        _, fragment_graph = self._build("CO")
        self.assertEqual(fragment_graph.num_fragments, 1)
        self.assertEqual(fragment_graph.connection_index, ())

    def test_fragment_metadata_preserves_atoms_and_connections(self):
        from formula_design.fragment import build_fragment_metadata

        mol, fragment_graph = self._build("CCOC(=O)OCC")
        metadata = build_fragment_metadata(mol, fragment_graph)
        atom_indices = [
            atom_idx for fragment in metadata["fragments"]
            for atom_idx in fragment["atom_indices"]
        ]
        self.assertEqual(sorted(atom_indices), list(range(mol.GetNumAtoms())))
        self.assertEqual(len(metadata["connections"]),
                         len(fragment_graph.cut_bond_indices))
        for fragment in metadata["fragments"]:
            self.assertIn("smiles", fragment)
            self.assertIn("atom_map_numbers", fragment)

    def test_fragment_attention_is_attached_in_fragment_order(self):
        from formula_design.fragment import attach_fragment_attention

        metadata = [[{
            "molecule_name": "solvent",
            "mapped_smiles": "[CH3:1][O:2][CH3:3]",
            "formal_charge": 0,
            "fragments": [{
                "fragment_id": 0
            }, {
                "fragment_id": 1
            }],
            "connections": [{
                "fragment_ids": [0, 1]
            }],
        }]]
        output = attach_fragment_attention(metadata, [0.25, 0.75], True)
        fragments = output[0][0]["fragments"]
        self.assertEqual(fragments[0]["pooling_attention"], 0.25)
        self.assertEqual(fragments[1]["pooling_attention"], 0.75)
        self.assertTrue(output[0][0]["fragment_encoder_applied"])


if __name__ == "__main__":
    unittest.main()
