import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "analysis"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "rules"))

import utils
import analyze
import summary_plots


def test_state_to_family_mapping():
    assert utils.state_to_family("TssA") == "Promoter"
    assert utils.state_to_family("1_TssA") == "Promoter"
    assert utils.state_to_family("TssAFlnk") == "Promoter"
    assert utils.state_to_family("Enh") == "Enhancer"
    assert utils.state_to_family("EnhG") == "Enhancer"
    assert utils.state_to_family("Tx") == "Transcribed"
    assert utils.state_to_family("TxWk") == "Transcribed"
    assert utils.state_to_family("TssBiv") == "Bivalent"
    assert utils.state_to_family("ReprPC") == "Polycomb"
    assert utils.state_to_family("Het") == "Heterochromatin"
    assert utils.state_to_family("ZNF/Rpts") == "Heterochromatin"
    assert utils.state_to_family("Quies") == "Quiescent"


def test_compute_pair_state_overlap_matrix():
    # Synthetic BED segments for annotations i and j
    # chrom, start, end, state
    segs_i = [
        ("chr1", 0, 100, "TssA"),
        ("chr1", 100, 300, "Tx"),
        ("chr1", 300, 500, "Quies"),
    ]
    segs_j = [
        ("chr1", 0, 50, "TssA"),
        ("chr1", 50, 150, "Tx"),
        ("chr1", 150, 300, "Tx"),
        ("chr1", 300, 500, "Quies"),
    ]

    # C_ij(A, B) = bp assigned A in i and B in j
    C = analyze.compute_pair_state_overlap_matrix(segs_i, segs_j, states=["TssA", "Tx", "Quies"])
    assert isinstance(C, pd.DataFrame)
    assert C.loc["TssA", "TssA"] == 50   # 0..50
    assert C.loc["TssA", "Tx"] == 50     # 50..100
    assert C.loc["Tx", "Tx"] == 200      # 100..300
    assert C.loc["Quies", "Quies"] == 200 # 300..500
    assert C.loc["Quies", "TssA"] == 0

    # Family mapping
    C_fam = analyze.compute_pair_state_overlap_matrix(segs_i, segs_j, map_to_family=True)
    assert isinstance(C_fam, pd.DataFrame)
    assert list(C_fam.index) == utils.STATE_FAMILY_ORDER
    assert C_fam.loc["Promoter", "Promoter"] == 50
    assert C_fam.loc["Promoter", "Transcribed"] == 50
    assert C_fam.loc["Transcribed", "Transcribed"] == 200
    assert C_fam.loc["Quiescent", "Quiescent"] == 200


def test_symmetrize_matrix():
    C = pd.DataFrame([
        [100, 20],
        [10, 50]
    ], index=["A", "B"], columns=["A", "B"])
    S = analyze.symmetrize_state_overlap_matrix(C)
    expected = pd.DataFrame([
        [200, 30],
        [30, 100]
    ], index=["A", "B"], columns=["A", "B"])
    pd.testing.assert_frame_equal(S, expected)

    # Numpy array
    C_arr = np.array([[100, 20], [10, 50]])
    S_arr = analyze.symmetrize_state_overlap_matrix(C_arr)
    np.testing.assert_array_equal(S_arr, np.array([[200, 30], [30, 100]]))


def test_aggregate_state_switch_matrices_pooled():
    S1 = pd.DataFrame([[10, 2], [2, 20]], index=["A", "B"], columns=["A", "B"])
    S2 = pd.DataFrame([[5, 1], [1, 10]], index=["A", "B"], columns=["A", "B"])

    M_pooled = analyze.aggregate_state_switch_matrices([S1, S2], mode="pooled")
    expected = pd.DataFrame([[15, 3], [3, 30]], index=["A", "B"], columns=["A", "B"])
    pd.testing.assert_frame_equal(M_pooled, expected)


def test_aggregate_state_switch_matrices_balanced():
    # Two replicate groups: ESC (2 pairs), SKIN (1 pair)
    # ESC pair 1, ESC pair 2, SKIN pair 1
    S_esc1 = pd.DataFrame([[100, 20], [20, 200]], index=["A", "B"], columns=["A", "B"])
    S_esc2 = pd.DataFrame([[120, 40], [40, 220]], index=["A", "B"], columns=["A", "B"])
    S_skin = pd.DataFrame([[50, 10], [10, 80]], index=["A", "B"], columns=["A", "B"])

    pair_matrices = {
        ("E001", "E002"): S_esc1,
        ("E001", "E003"): S_esc2,
        ("E055", "E056"): S_skin,
    }
    sample_groups = {
        "E001": "ESC", "E002": "ESC", "E003": "ESC",
        "E055": "SKIN", "E056": "SKIN"
    }

    # Mean ESC: [[110, 30], [30, 210]]
    # Mean SKIN: [[50, 10], [10, 80]]
    # Overall mean: [[80, 20], [20, 145]]
    M_balanced = analyze.aggregate_state_switch_matrices(pair_matrices, mode="balanced", sample_groups=sample_groups)
    expected = pd.DataFrame([[80.0, 20.0], [20.0, 145.0]], index=["A", "B"], columns=["A", "B"])
    pd.testing.assert_frame_equal(M_balanced, expected)


def test_compute_state_switch_probabilities():
    M = pd.DataFrame([
        [80, 20],
        [10, 90]
    ], index=["A", "B"], columns=["A", "B"])

    P = analyze.compute_state_switch_probabilities(M)
    # Row 1 sum = 100 -> P(A|A) = 0.8, P(B|A) = 0.2
    # Row 2 sum = 100 -> P(A|B) = 0.1, P(B|B) = 0.9
    expected = pd.DataFrame([
        [0.8, 0.2],
        [0.1, 0.9]
    ], index=["A", "B"], columns=["A", "B"])
    pd.testing.assert_frame_equal(P, expected)


def test_compute_state_switch_enrichment():
    # Within condition: highly stable (diagonal high)
    M_within = pd.DataFrame([
        [900, 100],
        [100, 900]
    ], index=["A", "B"], columns=["A", "B"])

    # Between condition: higher transition (off-diagonal increased)
    M_between = pd.DataFrame([
        [600, 400],
        [400, 600]
    ], index=["A", "B"], columns=["A", "B"])

    # P_within: [[0.9, 0.1], [0.1, 0.9]]
    # P_between: [[0.6, 0.4], [0.4, 0.6]]
    # E(A, A) = log10(0.6 / 0.9) = log10(2/3) ~ -0.17609
    # E(A, B) = log10(0.4 / 0.1) = log10(4) ~ 0.60206
    E = analyze.compute_state_switch_enrichment(M_within, M_between)

    assert pytest.approx(E.loc["A", "A"], 0.001) == np.log10(0.6 / 0.9)
    assert pytest.approx(E.loc["A", "B"], 0.001) == np.log10(0.4 / 0.1)
    assert pytest.approx(E.loc["B", "A"], 0.001) == np.log10(0.4 / 0.1)
    assert pytest.approx(E.loc["B", "B"], 0.001) == np.log10(0.6 / 0.9)


def test_fold_state_overlap_to_families():
    ov_dict = {
        ("TssA", "TssA"): 100,
        ("Tx", "TssA"): 50,
        ("Quies", "Quies"): 500,
    }
    df_fam = analyze.fold_state_overlap_to_families(ov_dict)
    assert list(df_fam.index) == utils.STATE_FAMILY_ORDER
    assert list(df_fam.columns) == utils.STATE_FAMILY_ORDER
    # Source is state in i (second item in pair_overlap key), dest is state in j (first item)
    assert df_fam.loc["Promoter", "Promoter"] == 100
    assert df_fam.loc["Promoter", "Transcribed"] == 50
    assert df_fam.loc["Quiescent", "Quiescent"] == 500


def test_plot_state_switch_heatmap_and_grid(tmp_path):
    fams = utils.STATE_FAMILY_ORDER
    # Generate dummy 7x7 matrix
    mat = pd.DataFrame(
        np.random.uniform(-0.5, 0.5, size=(len(fams), len(fams))),
        index=fams, columns=fams
    )
    # Set diagonal negative (depleted)
    for f in fams:
        mat.loc[f, f] = -0.4

    heatmap_path = tmp_path / "test_heatmap.pdf"
    summary_plots.plot_state_switch_heatmap(mat, title="Test Heatmap", path=str(heatmap_path))
    assert os.path.exists(heatmap_path)

    # Grid config
    configs = {
        "chromhmm_default": mat,
        "kmeans_homer": mat,
        "kmeans_macs2": mat,
        "kmeans_omni": mat,
        "joint_chromhmm": mat,
        "joint_kmeans_homer": mat,
        "joint_kmeans_macs2": mat,
        "joint_kmeans_omni": mat,
    }
    grid_path = tmp_path / "test_grid.pdf"
    summary_plots.plot_state_switch_grid(configs, path=str(grid_path))
    assert os.path.exists(grid_path)

    # Multi-dataset plot
    ds_dict = {
        "ENCODE": mat,
        "SAGAconf": mat,
        "1000 epigenomes": mat,
        "Aggregated": mat
    }
    ds_path = tmp_path / "test_datasets.pdf"
    summary_plots.plot_state_switch_datasets(ds_dict, method_label="ChromHMM default", path=str(ds_path))
    assert os.path.exists(ds_path)


def test_notebooks_contain_state_switch():
    import json
    for nb_name in ["analysis_epi1000.ipynb", "analysis_encode.ipynb", "analysis_sagaconf.ipynb", "summary.ipynb"]:
        nb_path = os.path.join(os.path.dirname(__file__), "..", nb_name)
        assert os.path.exists(nb_path), f"Notebook {nb_name} not found"
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        all_sources = "".join("".join(c.get("source", [])) for c in nb.get("cells", []))
        assert "State-switch analysis" in all_sources or "state-switch" in all_sources.lower(), f"Missing state-switch title in {nb_name}"
        assert "plot_state_switch_grid" in all_sources, f"Missing plot_state_switch_grid in {nb_name}"
        assert "state_switch_grid_balanced" in all_sources, f"Missing balanced grid plot in {nb_name}"
        if nb_name == "summary.ipynb":
            assert "state_switch_chromhmm_default" in all_sources, f"Missing state_switch_chromhmm_default in {nb_name}"
            assert "plot_state_switch_datasets" in all_sources, f"Missing plot_state_switch_datasets in {nb_name}"
        else:
            assert "plot_state_switch_heatmap" in all_sources, f"Missing plot_state_switch_heatmap in {nb_name}"
