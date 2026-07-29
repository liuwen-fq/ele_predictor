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


if __name__ == "__main__":
    unittest.main()
