"""Contracts for training-time interaction auxiliary supervision.

The mechanism auxiliary is deliberately *not* a third/fourth generation
condition.  These tests protect that distinction: even when an auxiliary
label is present, diffusion receives only the user-provided two-dimensional
``[conductivity, anion_ratio]`` condition.
"""

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - handled by the class skip below
    torch = None

if torch is not None:
    from formula_design.data import Data
    from formula_design.generator import FormulaDiffusion
    from formula_design.train.loss import LossType, loss_func


_BATCH_SIZE = 3
_EMBEDDING_DIM = 8


def _diffusion(*, mechanism_auxiliary=None):
    """Make a deliberately small CPU diffusion model for contract tests."""
    diff_config = {"time_dim": None}
    if mechanism_auxiliary is not None:
        diff_config["mechanism_auxiliary"] = mechanism_auxiliary
    return FormulaDiffusion(
        unet1d_config={
            "dim": 8,
            "prop_dim": 2,
            "dim_mults": [1],
            "channels": 1,
            "dropout": 0.0,
            "self_condition": False,
            "learned_variance": False,
            "learned_sinusoidal_cond": False,
            "random_fourier_features": False,
            "learned_sinusoidal_dim": 4,
            "sinusoidal_pos_emb_theta": 10000,
            "attn_dim_head": 4,
            "attn_heads": 2,
            "condition_method": "concat",
        },
        beta_scheduler_config={"timesteps": 4, "scheduler_mode": "linear"},
        sigma_scheduler_config={"timesteps": 4},
        diff_config=diff_config,
        embedding_dim=_EMBEDDING_DIM,
    )


def _batch(*, mechanism_label_shape=(_BATCH_SIZE, 2)):
    data = Data()
    data.frm_emb = torch.randn(_BATCH_SIZE, _EMBEDDING_DIM)
    data.conductivity = torch.tensor([0.1, 0.2, 0.3])
    data.anion_ratio = torch.tensor([0.35, 0.50, 0.65])
    data.mechanism_aux_emb = torch.randn(*mechanism_label_shape)
    return data


def _record_diffusion_condition(model):
    conditions = []

    def capture_condition(_module, args):
        # Unet1D.forward(time, x, context, ...)
        conditions.append(args[2].detach().clone())

    return conditions, model.noise_predictor.register_forward_pre_hook(
        capture_condition)


@unittest.skipIf(torch is None, "torch is not installed")
class MechanismAuxiliaryTest(unittest.TestCase):

    def test_auxiliary_uses_two_dimensional_property_condition_and_backpropagates(
            self):
        """MA labels supervise training without becoming diffusion conditions."""
        model = _diffusion(mechanism_auxiliary={
            "enabled": True,
            "label_dim": 2,
            "hidden_dim": 32,
        })
        data = _batch()
        seen_conditions, handle = _record_diffusion_condition(model)
        try:
            output = model(data)
        finally:
            handle.remove()

        self.assertTrue(seen_conditions)
        expected_condition = torch.stack(
            (data.conductivity, data.anion_ratio), dim=-1)
        for condition in seen_conditions:
            self.assertEqual(tuple(condition.shape), (_BATCH_SIZE, 2))
            self.assertTrue(torch.equal(condition, expected_condition))

        self.assertEqual(tuple(output["mechanism_aux_pred"].shape),
                         (_BATCH_SIZE, 2))
        self.assertEqual(tuple(output["mechanism_aux_snr_weight"].shape),
                         (_BATCH_SIZE, ))
        self.assertTrue(torch.isfinite(output["mechanism_aux_pred"]).all())
        self.assertTrue(
            torch.isfinite(output["mechanism_aux_snr_weight"]).all())

        auxiliary_loss = loss_func(output, data, LossType.mechanism_aux_Huber)
        self.assertEqual(auxiliary_loss.ndim, 0)
        self.assertTrue(torch.isfinite(auxiliary_loss))
        auxiliary_loss.backward()
        self.assertTrue(any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in model.parameters()))

    def test_auxiliary_rejects_non_vector_pair_labels(self):
        """The auxiliary target must be one (I_sigma, I_r) vector per formula."""
        model = _diffusion(mechanism_auxiliary={
            "enabled": True,
            "label_dim": 2,
            "hidden_dim": 32,
        })
        data = _batch(mechanism_label_shape=(_BATCH_SIZE, ))

        with self.assertRaisesRegex(
                ValueError, "mechanism auxiliary prediction/label shape mismatch"):
            loss_func(model(data), data, LossType.mechanism_aux_Huber)

    def test_single_target_auxiliary_keeps_two_dimensional_generation_condition(
            self):
        """σ-only or r-only MA uses a one-value label, never a new condition."""
        model = _diffusion(mechanism_auxiliary={
            "enabled": True,
            "label_dim": 1,
            "hidden_dim": 32,
        })
        data = _batch(mechanism_label_shape=(_BATCH_SIZE, 1))
        seen_conditions, handle = _record_diffusion_condition(model)
        try:
            output = model(data)
        finally:
            handle.remove()

        self.assertTrue(all(tuple(condition.shape) == (_BATCH_SIZE, 2)
                            for condition in seen_conditions))
        self.assertEqual(tuple(output["mechanism_aux_pred"].shape),
                         (_BATCH_SIZE, 1))
        loss = loss_func(output, data, LossType.mechanism_aux_Huber)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_two_dimensional_diffusion_stays_backward_compatible_without_auxiliary_head(
            self):
        """Existing G0/G2-Q runs remain ordinary two-condition diffusion jobs."""
        model = _diffusion()
        data = _batch()
        seen_conditions, handle = _record_diffusion_condition(model)
        try:
            output = model(data)
        finally:
            handle.remove()

        self.assertTrue(seen_conditions)
        self.assertTrue(all(tuple(condition.shape) == (_BATCH_SIZE, 2)
                            for condition in seen_conditions))
        self.assertNotIn("mechanism_aux_pred", output)
        self.assertNotIn("mechanism_aux_snr_weight", output)

        diffusion_loss = loss_func(output, data, LossType.diff_MSE)
        self.assertEqual(diffusion_loss.ndim, 0)
        diffusion_loss.backward()
        self.assertTrue(any(parameter.grad is not None
                            for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
