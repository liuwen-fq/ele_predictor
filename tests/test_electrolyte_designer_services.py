import unittest

from apps.electrolyte_designer.services.catalog_service import (
    automatic_selection,
    catalogue_summary,
    filter_catalogue,
    property_constraints,
)
from apps.electrolyte_designer.services.data_access import default_paths, load_metric_rows
from apps.electrolyte_designer.services.generation_service import rank_frozen_samples


class ElectrolyteDesignerServicesTest(unittest.TestCase):
    def test_demo_metrics_are_available(self):
        rows = load_metric_rows(default_paths())
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(0.0 <= row["r2"] <= 1.0 for row in rows))

    def test_gdb17_catalogue_can_be_filtered_and_selected(self):
        paths = default_paths()
        summary = catalogue_summary(str(paths.root), str(paths.gdb17_catalogue))
        self.assertGreaterEqual(len(summary), 62)
        constraints = property_constraints(paths)
        subset = filter_catalogue(
            summary,
            (),
            (0.0, 100.0),
            (0.0, 12.0),
            (-100.0, 150.0),
        )
        self.assertGreaterEqual(len(subset), 62)
        selected = automatic_selection(
            subset[:128],
            constraints,
            paths.root,
            pool_size=8,
            seed=7,
        )
        self.assertEqual(len(selected), 8)
        self.assertEqual(
            [row["selected_step"] for row in selected],
            list(range(1, 9)),
        )

    def test_existing_generated_samples_can_be_reranked(self):
        from apps.electrolyte_designer.services.data_access import (
            load_generation_assets,
        )

        records, solvents, salts = load_generation_assets(default_paths())
        ranked = rank_frozen_samples(
            records,
            solvents,
            salts,
            20.0,
            0.5,
            2.0,
            0.05,
        )
        self.assertTrue(ranked)
        self.assertLessEqual(ranked[0]["joint_loss"], ranked[-1]["joint_loss"])
        self.assertTrue(ranked[0]["solvents"])


if __name__ == "__main__":
    unittest.main()
