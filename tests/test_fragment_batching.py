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

    def test_sidecar_mode_preserves_graph_states_exactly(self):
        from formula_design.predictor.fragment_block import FragmentBlock

        data = self._sample(2, [0, 1], [(1, 0), (0, 1)])
        node_h = torch.randn(2, 8)
        edge_h = torch.randn(2, 6)
        block = FragmentBlock(
            node_dim=8,
            edge_dim=6,
            hidden_dim=12,
            mode="sidecar",
        )
        output_node_h, output_edge_h = block(data, node_h, edge_h)

        self.assertTrue(torch.equal(output_node_h, node_h))
        self.assertTrue(torch.equal(output_edge_h, edge_h))
        self.assertEqual(block.get_molecule_embedding().shape, (1, 12))

    def test_sidecar_zero_gate_is_an_exact_identity(self):
        from formula_design.predictor.fragment_block import FragmentBlock

        data = self._sample(2, [0, 1], [(1, 0), (0, 1)])
        node_h = torch.randn(2, 8)
        edge_h = torch.randn(2, 6)
        block = FragmentBlock(
            node_dim=8,
            edge_dim=6,
            hidden_dim=12,
            mode="sidecar",
            sidecar_gate_init=0.0,
        )
        block(data, node_h, edge_h)
        base_embedding = torch.randn(1, 14)

        self.assertTrue(
            torch.equal(block.fuse_molecule_embedding(base_embedding),
                        base_embedding))

    def test_formula_sidecar_zero_gates_preserve_base_representation(self):
        from formula_design.data import Data
        from formula_design.predictor.aggr import FragmentFormulaAggr

        data = Data()
        data.counts_mol = torch.tensor([[2, 2], [1, 1]], dtype=torch.int32)
        data.molar_ratios = torch.tensor([[0.4], [0.6], [1.0]])
        molecule_h = torch.randn(3, 12)
        base_h = torch.randn(2, 16)
        aggr = FragmentFormulaAggr(
            input_dim=12,
            hidden_dim=10,
            output_dim=16,
            gate_init=0.0,
        )

        h_ne, h_pred = aggr(data, molecule_h, base_h)
        self.assertTrue(torch.equal(h_ne, base_h))
        self.assertTrue(torch.equal(h_pred, base_h))

    def test_contribution_head_is_zero_at_initialization(self):
        from formula_design.data import Data
        from formula_design.predictor.aggr import (
            ConditionAwareFragmentContribution,
        )

        data = Data()
        data.counts_mol = torch.tensor([[2, 2], [1, 1]], dtype=torch.int32)
        data.molar_ratios = torch.tensor([[0.4], [0.6], [1.0]])
        fragment_h = torch.randn(5, 12)
        fragment_batch = torch.tensor([0, 0, 1, 1, 2])
        condition_h = torch.randn(2, 10)
        head = ConditionAwareFragmentContribution(
            input_dim=12,
            condition_input_dim=10,
            hidden_dim=8,
            main_rank=4,
            pair_rank=3,
        )

        output = head(data, fragment_h, fragment_batch, condition_h)
        for task in head.task_names:
            task_output = output["tasks"][task]
            self.assertTrue(torch.equal(
                task_output["residual"], torch.zeros(2)))
            self.assertTrue(torch.equal(
                task_output["main_sum"], torch.zeros(2)))
            self.assertTrue(torch.equal(
                task_output["interaction_sum"], torch.zeros(2)))

    def test_contribution_decomposition_and_cross_molecule_pairs(self):
        from formula_design.data import Data
        from formula_design.predictor.aggr import (
            ConditionAwareFragmentContribution,
        )

        data = Data()
        data.counts_mol = torch.tensor([[2, 2]], dtype=torch.int32)
        data.molar_ratios = torch.tensor([[0.4], [0.6]])
        fragment_h = torch.randn(3, 12)
        fragment_batch = torch.tensor([0, 0, 1])
        condition_h = torch.randn(1, 10)
        head = ConditionAwareFragmentContribution(
            input_dim=12,
            condition_input_dim=10,
            hidden_dim=8,
            main_rank=4,
            pair_rank=3,
            cross_molecule_only=True,
            diagnostic_top_k_pairs=10,
        )
        with torch.no_grad():
            for projection in head.main_condition.values():
                projection.bias.fill_(0.5)
            for projection in head.pair_condition.values():
                projection.bias.fill_(0.5)
        head.eval()

        output = head(data, fragment_h, fragment_batch, condition_h)
        task_output = output["tasks"]["conductivity_mistry"]
        self.assertTrue(torch.allclose(
            task_output["residual"],
            task_output["main_sum"] + task_output["interaction_sum"],
        ))
        # Two fragments in molecule 0 interact only with the one fragment in
        # molecule 1.  The within-molecule pair is deliberately excluded.
        self.assertEqual(task_output["pair_indices"].shape[0], 2)
        self.assertTrue(torch.allclose(
            task_output["pair_effect"].sum(),
            task_output["interaction_sum"][0],
            atol=1.0e-7,
        ))
        pair_indices = task_output["pair_indices"]
        for left, right in pair_indices.tolist():
            self.assertNotEqual(fragment_batch[left], fragment_batch[right])

    def test_contribution_normalizes_fragments_and_bounds_residual(self):
        from formula_design.data import Data
        from formula_design.predictor.aggr import (
            ConditionAwareFragmentContribution,
        )

        data = Data()
        data.counts_mol = torch.tensor([[2, 2]], dtype=torch.int32)
        data.molar_ratios = torch.tensor([[0.4], [0.6]])
        fragment_batch = torch.tensor([0, 0, 1])
        head = ConditionAwareFragmentContribution(
            input_dim=12,
            condition_input_dim=10,
            hidden_dim=8,
            main_rank=4,
            pair_rank=3,
            residual_bound=0.02,
            diagnostic_top_k_pairs=-1,
        )
        with torch.no_grad():
            for projection in head.main_condition.values():
                projection.bias.fill_(100.0)
            for projection in head.pair_condition.values():
                projection.bias.fill_(100.0)
        head.eval()
        output = head(
            data,
            torch.randn(3, 12),
            fragment_batch,
            torch.randn(1, 10),
        )
        self.assertTrue(torch.allclose(
            output["fragment_normalization_count"],
            torch.tensor([2.0, 2.0, 1.0]),
        ))
        self.assertTrue(torch.allclose(
            output["fragment_weight"],
            torch.tensor([0.2, 0.2, 0.6]),
        ))
        task = output["tasks"]["conductivity_mistry"]
        self.assertLessEqual(float(task["residual"].abs().max()), 0.02)
        self.assertTrue(torch.allclose(
            task["main_sum"] + task["interaction_sum"],
            task["residual"],
            atol=1.0e-7,
        ))
        self.assertTrue(torch.allclose(
            task["pair_effect"].sum(), task["interaction_sum"].sum(),
            atol=1.0e-7,
        ))

    def test_contribution_head_receives_gradient_from_zero_residual(self):
        from formula_design.data import Data
        from formula_design.predictor.aggr import (
            ConditionAwareFragmentContribution,
        )

        data = Data()
        data.counts_mol = torch.tensor([[2, 2]], dtype=torch.int32)
        data.molar_ratios = torch.tensor([[0.5], [0.5]])
        head = ConditionAwareFragmentContribution(
            input_dim=12,
            condition_input_dim=10,
            hidden_dim=8,
            main_rank=4,
            pair_rank=3,
        )
        output = head(
            data,
            torch.randn(3, 12),
            torch.tensor([0, 0, 1]),
            torch.randn(1, 10),
        )
        residual = output["tasks"]["conductivity_mistry"]["residual"]
        (residual - 1.0).square().mean().backward()

        main_grad = head.main_condition[
            "conductivity_mistry"].weight.grad
        pair_grad = head.pair_condition[
            "conductivity_mistry"].weight.grad
        self.assertGreater(float(main_grad.abs().sum()), 0.0)
        self.assertGreater(float(pair_grad.abs().sum()), 0.0)

    def test_explanation_head_adds_condition_reference_and_fragments(self):
        from formula_design.data import Data
        from formula_design.predictor.aggr import (
            ConditionAwareFragmentExplainer,
        )

        data = Data()
        data.counts_mol = torch.tensor([[2, 2]], dtype=torch.int32)
        data.molar_ratios = torch.tensor([[0.4], [0.6]])
        head = ConditionAwareFragmentExplainer(
            input_dim=12,
            condition_input_dim=10,
            hidden_dim=8,
            main_rank=4,
            pair_rank=3,
            cross_molecule_only=True,
            diagnostic_top_k_pairs=10,
        )
        with torch.no_grad():
            head.condition_reference["conductivity_explain"].bias.fill_(0.3)
            head.main_condition["conductivity_explain"].bias.fill_(0.5)
            head.pair_condition["conductivity_explain"].bias.fill_(0.5)
        head.eval()

        output = head(
            data,
            torch.randn(3, 12),
            torch.tensor([0, 0, 1]),
            torch.randn(1, 10),
        )
        task = output["tasks"]["conductivity_explain"]
        self.assertTrue(torch.allclose(
            task["explanation_prediction"],
            task["condition_reference"] + task["main_sum"]
            + task["interaction_sum"],
            atol=1.0e-7,
        ))
        self.assertTrue(torch.allclose(
            task["pair_effect"].sum(), task["interaction_sum"].sum(),
            atol=1.0e-7,
        ))

    def test_contribution_head_integrates_with_formula_model(self):
        from formula_design.data import FormulaData, collate_data
        from formula_design.mol import Molecule
        from formula_design.predictor import MolMix, Mono

        def mapped(smiles):
            return Molecule.from_smiles(smiles).get_mapped_smiles()

        def formula(first_solvent, second_solvent):
            return FormulaData(
                ["solvent_a", "solvent_b"],
                [mapped(first_solvent), mapped(second_solvent)],
                ["LiPF6"],
                [mapped("F[P-](F)(F)(F)(F)F")],
                [0.4, 0.6],
                [1.0],
                concentration=0.1,
                temperature=0.8,
            )

        graph = collate_data([
            formula("CCOC(=O)OCC", "CC#N"),
            formula("O=C1OCCO1", "COC(=O)OC"),
        ])
        pretrained = Mono(
            graph_block={
                "feature_layer": {
                    "atom_embedding_dim": 16,
                    "node_mlp_dims": [32, 32, 2],
                    "edge_mlp_dims": [32, 32, 2],
                    "act": "gelu",
                },
                "gnn_layer": {
                    "gnn_type": "EGT",
                    "gnn_dims": [32, 32, 3],
                    "jk": "cat",
                    "act": "gelu",
                    "heads": 4,
                    "at_channels": 8,
                    "ffn_dims": [32, 2],
                },
                "fragment_block": {
                    "enabled": True,
                    "mode": "sidecar",
                    "freeze_base": True,
                    "hidden_dim": 64,
                },
            },
            readout_block={
                "input_dim": 64,
                "hidden_dims": [16],
                "output_dim": 1,
            },
        )
        model = MolMix(
            pretrained,
            aggr_block={
                "node_emb_dim": 32,
                "node_att_dim": 32,
                "edge_emb_dim": 32,
                "edge_att_dim": 32,
            },
            readout_block={
                "input_dim": 384,
                "hidden_dims": [16],
                "output_dim": 1,
            },
            anion_block={
                "input_dim": 400,
                "hidden_dims": [16],
                "output_dim": 1,
            },
            train_mode="finetune",
            fragment_contribution_block={
                "enabled": True,
                "input_dim": 64,
                "condition_input_dim": 130,
                "hidden_dim": 16,
                "main_rank": 4,
                "pair_rank": 3,
                "freeze_base_formula": True,
            },
        )
        model.eval()
        trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable)
        self.assertTrue(all(
            name.startswith("fragment_contribution_head.")
            for name in trainable))
        output = model(graph)

        self.assertTrue(torch.equal(output["conductivity"],
                                    output["conductivity_base"]))
        self.assertEqual(output["conductivity"].shape, (2, 1))
        contributions = output["fragment_contributions"]
        self.assertEqual(
            contributions["tasks"]["conductivity_mistry"]["residual"].shape,
            (2, ),
        )
        self.assertEqual(contributions["fragment_formula_index"].dim(), 1)

    def test_explanation_head_preserves_bamboo_prediction(self):
        from formula_design.data import FormulaData, collate_data
        from formula_design.mol import Molecule
        from formula_design.predictor import MolMix, Mono

        def mapped(smiles):
            return Molecule.from_smiles(smiles).get_mapped_smiles()

        graph = collate_data([FormulaData(
            ["solvent_a", "solvent_b"],
            [mapped("CCOC(=O)OCC"), mapped("CC#N")],
            ["LiPF6"], [mapped("F[P-](F)(F)(F)(F)F")],
            [0.4, 0.6], [1.0], concentration=0.1, temperature=0.8,
        )])
        pretrained = Mono(
            graph_block={
                "feature_layer": {
                    "atom_embedding_dim": 16,
                    "node_mlp_dims": [32, 32, 2],
                    "edge_mlp_dims": [32, 32, 2],
                    "act": "gelu",
                },
                "gnn_layer": {
                    "gnn_type": "EGT", "gnn_dims": [32, 32, 3],
                    "jk": "cat", "act": "gelu", "heads": 4,
                    "at_channels": 8, "ffn_dims": [32, 2],
                },
                "fragment_block": {
                    "enabled": True, "mode": "sidecar",
                    "freeze_base": True, "hidden_dim": 64,
                },
            },
            readout_block={"input_dim": 64, "hidden_dims": [16],
                           "output_dim": 1},
        )
        model = MolMix(
            pretrained,
            aggr_block={"node_emb_dim": 32, "node_att_dim": 32,
                        "edge_emb_dim": 32, "edge_att_dim": 32},
            readout_block={"input_dim": 384, "hidden_dims": [16],
                          "output_dim": 1},
            anion_block={"input_dim": 400, "hidden_dims": [16],
                         "output_dim": 1},
            train_mode="finetune",
            fragment_explanation_block={
                "enabled": True, "input_dim": 64,
                "condition_input_dim": 130, "hidden_dim": 16,
                "main_rank": 4, "pair_rank": 3,
                "freeze_base_formula": True,
            },
        )
        model.eval()
        trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable)
        self.assertTrue(all(
            name.startswith("fragment_explanation_head.")
            for name in trainable))

        output = model(graph)
        self.assertTrue(torch.equal(output["conductivity"],
                                    output["conductivity_base"]))
        self.assertEqual(output["conductivity"].shape, (1, 1))
        task = output["fragment_explanations"]["tasks"][
            "conductivity_explain"]
        self.assertTrue(torch.allclose(
            output["conductivity_explain"].squeeze(),
            task["condition_reference"] + task["main_sum"]
            + task["interaction_sum"],
            atol=1.0e-7,
        ))

    def test_prediction_serializes_fragment_and_pair_effects(self):
        from scripts.test_results.predict import collect_fragment_contributions

        metadata = [[{
            "molecule_name": "A",
            "fragments": [{"fragment_id": 0, "smiles": "CC"}],
        }, {
            "molecule_name": "B",
            "fragments": [{"fragment_id": 0, "smiles": "C#N"}],
        }]]
        task = {
            "main_effect": torch.tensor([0.01, -0.02]),
            "main_sum": torch.tensor([-0.01]),
            "interaction_sum": torch.tensor([0.03]),
            "residual": torch.tensor([0.02]),
            "pair_indices": torch.tensor([[0, 1]]),
            "pair_formula_index": torch.tensor([0]),
            "pair_effect": torch.tensor([0.03]),
        }
        diagnostics = {
            "contributions": {
                "tasks": {
                    "conductivity_mistry": task,
                },
            },
        }

        output = collect_fragment_contributions(
            None, metadata, diagnostics)
        self.assertAlmostEqual(
            metadata[0][0]["fragments"][0][
                "conductivity_main_effect_normalized"],
            0.01,
            places=6,
        )
        self.assertEqual(len(output[0]["top_fragment_pairs"]), 1)
        self.assertAlmostEqual(output[0]["total_residual_normalized"],
                               0.02,
                               places=6)
        self.assertGreater(output[0]["conductivity_multiplier"], 1.0)

    def test_prediction_serializes_independent_explanation(self):
        from scripts.test_results.predict import collect_fragment_contributions

        metadata = [[{
            "molecule_name": "A",
            "fragments": [{"fragment_id": 0, "smiles": "CC"}],
        }]]
        diagnostics = {
            "explanations": {
                "tasks": {
                    "conductivity_explain": {
                        "main_effect": torch.tensor([0.02]),
                        "main_sum": torch.tensor([0.02]),
                        "interaction_sum": torch.tensor([0.01]),
                        "residual": torch.tensor([0.03]),
                        "condition_reference": torch.tensor([0.4]),
                        "explanation_prediction": torch.tensor([0.43]),
                    },
                },
            },
        }

        output = collect_fragment_contributions(
            None,
            metadata,
            diagnostics,
            task="conductivity_explain",
            diagnostic_key="explanations",
        )
        self.assertAlmostEqual(
            output[0]["condition_reference_normalized"], 0.4, places=6)
        self.assertAlmostEqual(
            output[0]["explanation_prediction_normalized"], 0.43,
            places=6)
        self.assertAlmostEqual(
            metadata[0][0]["fragments"][0][
                "conductivity_main_effect_normalized"], 0.02, places=6)

    def test_freeze_base_only_leaves_fragment_parameters_trainable(self):
        from formula_design.predictor import Mono

        model = Mono(
            graph_block={
                "feature_layer": {
                    "atom_embedding_dim": 16,
                    "node_mlp_dims": [32, 32, 2],
                    "edge_mlp_dims": [32, 32, 2],
                    "act": "gelu",
                },
                "gnn_layer": {
                    "gnn_type": "EGT",
                    "gnn_dims": [32, 32, 3],
                    "jk": "cat",
                    "act": "gelu",
                    "heads": 4,
                    "at_channels": 8,
                    "ffn_dims": [32, 2],
                },
                "fragment_block": {
                    "enabled": True,
                    "mode": "sidecar",
                    "freeze_base": True,
                    "hidden_dim": 64,
                },
            },
            readout_block={
                "input_dim": 64,
                "hidden_dims": [16],
                "output_dim": 1,
            },
        )
        trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]

        self.assertTrue(trainable)
        self.assertTrue(
            all(
                name.startswith("graph_embed_block.fragment_block.")
                for name in trainable))


if __name__ == "__main__":
    unittest.main()
