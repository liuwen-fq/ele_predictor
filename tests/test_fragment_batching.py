import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class FragmentBatchingTest(unittest.TestCase):

    @staticmethod
    def _sample(fragment_count, atom_to_fragment, connections):
        from formula_design.data import Data

        data = Data()
        data.counts = torch.tensor(
            [[len(atom_to_fragment), 2, fragment_count]],
            dtype=torch.int32,
        )
        data.inc_node_edge = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
        data.node_features = torch.zeros((len(atom_to_fragment), 5),
                                         dtype=torch.int32)
        data.inc_fragment_atom = torch.tensor(atom_to_fragment,
                                              dtype=torch.int32)
        data.fragment_attachment = torch.zeros((len(atom_to_fragment), 1),
                                               dtype=torch.int32)
        data.inc_fragment_connection = torch.tensor(connections,
                                                    dtype=torch.int32).reshape(
                                                        -1, 2)
        data.fragment_connection_features = torch.zeros((len(connections), 9),
                                                        dtype=torch.int32)
        return data

    def test_fragment_indices_are_offset_during_collation(self):
        from formula_design.data import collate_data

        first = self._sample(2, [0, 1], [(1, 0), (0, 1)])
        second = self._sample(1, [0, 0], [])
        batch = collate_data([first, second])
        self.assertEqual(batch.inc_fragment_atom.tolist(), [0, 1, 2, 2])
        self.assertEqual(batch.inc_fragment_connection.tolist(),
                         [[1, 0], [0, 1]])

    def test_fragment_block_preserves_node_and_edge_shapes(self):
        from formula_design.data import collate_data
        from formula_design.predictor.fragment_block import FragmentBlock

        first = self._sample(2, [0, 1], [(1, 0), (0, 1)])
        second = self._sample(1, [0, 0], [])
        batch = collate_data([first, second])
        node_h = torch.randn(4, 8)
        edge_h = torch.randn(4, 6)
        block = FragmentBlock(node_dim=8, edge_dim=6, hidden_dim=12)
        output_node_h, output_edge_h = block(batch, node_h, edge_h)

        self.assertEqual(output_node_h.shape, node_h.shape)
        self.assertEqual(output_edge_h.shape, edge_h.shape)
        self.assertTrue(torch.isfinite(output_node_h).all())
        self.assertTrue(torch.isfinite(output_edge_h).all())

    def test_neutral_only_mode_bypasses_charged_molecule(self):
        from formula_design.predictor.fragment_block import FragmentBlock

        data = self._sample(1, [0, 0], [])
        data.node_features[0, 2] = -1
        node_h = torch.randn(2, 8)
        edge_h = torch.randn(2, 6)
        block = FragmentBlock(
            node_dim=8,
            edge_dim=6,
            hidden_dim=12,
            neutral_only=True,
        )
        output_node_h, output_edge_h = block(data, node_h, edge_h)
        self.assertTrue(torch.equal(output_node_h, node_h))
        self.assertTrue(torch.equal(output_edge_h, edge_h))


if __name__ == "__main__":
    unittest.main()
