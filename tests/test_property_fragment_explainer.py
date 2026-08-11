from types import SimpleNamespace

import torch

from formula_design.predictor.aggr import ConditionAwarePropertyFragmentExplainer


def test_property_fragment_decomposition_is_additive():
    data = SimpleNamespace(
        counts_mol=torch.tensor([[2, 0], [1, 0]]),
        molar_ratios=torch.tensor([[0.6], [0.4], [1.0]]),
    )
    model = ConditionAwarePropertyFragmentExplainer(
        input_dim=4,
        condition_input_dim=5,
        hidden_dim=8,
        main_rank=3,
        pair_rank=2,
        property_names=("dc", "vis", "Tm"),
        property_centers=(0.5, 0.5, 0.5),
        property_scales=(0.25, 0.25, 0.25),
        diagnostic_top_k_pairs=-1,
    )
    with torch.no_grad():
        model.property_condition["conductivity_explain"].weight.fill_(0.2)
    model.eval()
    output = model(
        data,
        torch.randn(3, 4),
        torch.tensor([0, 0, 2]),
        torch.randn(2, 5),
        torch.tensor([[0.2, 0.3, 0.8], [0.7, 0.4, 0.2], [0.6, 0.8, 0.1]]),
        fragment_active=torch.ones(3),
    )
    task = output["tasks"]["conductivity_explain"]
    assert torch.allclose(task["main_sum"] + task["interaction_sum"],
                          task["residual"], atol=1.0e-6)
    assert torch.allclose(task["property_by_name"].sum(dim=-1),
                          task["property_sum"], atol=1.0e-6)
    assert torch.allclose(task["condition_reference"] + task["property_sum"]
                          + task["residual"], task["explanation_prediction"],
                          atol=1.0e-6)
    assert task["property_molecule_effect"].shape == (3, 3)
