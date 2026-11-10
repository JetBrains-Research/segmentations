import csv
import gzip
import os
import tempfile
import pytest
import pandas as pd

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "rules")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "analysis")))

import analyze
import utils


def write_rnaseq_tsv(path, rows):
    """Helper to write RNA-seq TSV fixture."""
    fieldnames = ["gene_id", "transcript_id(s)", "length", "effective_length",
                  "expected_count", "TPM", "FPKM", "IsoPct"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_gtf(path, entries, gzipped=False):
    """Helper to write GTF fixture."""
    lines = ["# Synthetic GTF header\n"]
    for e in entries:
        attrs = f'gene_id "{e["gene_id"]}"; gene_name "{e["gene_name"]}"; gene_type "protein_coding";'
        line = f'{e["chrom"]}\tHAVANA\tgene\t{e["start"]}\t{e["end"]}\t.\t{e["strand"]}\t.\t{attrs}\n'
        lines.append(line)
    
    if gzipped:
        with gzip.open(path, "wt") as f:
            f.writelines(lines)
    else:
        with open(path, "w") as f:
            f.writelines(lines)


def test_load_gene_tpms_ranking_and_top_n():
    with tempfile.TemporaryDirectory() as tmpdir:
        rnaseq_file = os.path.join(tmpdir, "rnaseq.tsv")
        rows = [
            {"gene_id": "ENSG00000001.1", "TPM": "50.5"},
            {"gene_id": "ENSG00000002.2", "TPM": "10.0"},
            {"gene_id": "ENSG00000003.1", "TPM": "100.2"},
            {"gene_id": "ENSG00000004.5", "TPM": "0.0"},
            {"gene_id": "ENSG00000005.1", "TPM": "0.05"},
            {"gene_id": "INVALID", "TPM": "N/A"},
        ]
        write_rnaseq_tsv(rnaseq_file, rows)

        # Select top 2 genes
        tpms, top_exp = analyze.load_gene_tpms(rnaseq_file, top_n=2)
        # Top 2 should be ENSG00000003.1 (100.2) and ENSG00000001.1 (50.5)
        assert "ENSG00000003.1" in top_exp
        assert "ENSG00000003" in top_exp
        assert "ENSG00000001.1" in top_exp
        assert "ENSG00000001" in top_exp
        # ENSG00000002.2 ranked 3rd so not in top 2
        assert "ENSG00000002.2" not in top_exp
        assert "ENSG00000002" not in top_exp
        # Zero TPM gene must never be in top expressed
        assert "ENSG00000004.5" not in top_exp

        # Check TPM lookup map
        assert tpms["ENSG00000003.1"] == pytest.approx(100.2)
        assert tpms["ENSG00000003"] == pytest.approx(100.2)
        assert tpms["ENSG00000004.5"] == pytest.approx(0.0)


def test_load_gene_tpms_zero_tpm_cutoff():
    with tempfile.TemporaryDirectory() as tmpdir:
        rnaseq_file = os.path.join(tmpdir, "rnaseq.tsv")
        rows = [
            {"gene_id": "ENSG00000001.1", "TPM": "1.5"},
            {"gene_id": "ENSG00000002.1", "TPM": "0.0"},
        ]
        write_rnaseq_tsv(rnaseq_file, rows)

        # Requesting top 12k when only 1 gene has TPM > 0
        tpms, top_exp = analyze.load_gene_tpms(rnaseq_file, top_n=12000)
        assert "ENSG00000001.1" in top_exp
        assert "ENSG00000001" in top_exp
        assert "ENSG00000002.1" not in top_exp
        assert "ENSG00000002" not in top_exp


def test_load_gene_tpms_ranks_only_annotated_genes():
    with tempfile.TemporaryDirectory() as tmpdir:
        rnaseq_file = os.path.join(tmpdir, "rnaseq.tsv")
        rows = [
            {"gene_id": "gSpikein_ERCC-00130", "TPM": "900.0"},
            {"gene_id": "ENSG00000009.3", "TPM": "500.0"},   # retired from the GTF
            {"gene_id": "ENSG00000001.1", "TPM": "50.0"},
            {"gene_id": "ENSG00000002.1", "TPM": "10.0"},
        ]
        write_rnaseq_tsv(rnaseq_file, rows)

        # The GTF knows the genes under another version: matched on the base ID.
        known = {"ENSG00000001.4", "ENSG00000001", "ENSG00000002.7", "ENSG00000002"}
        tpms, top_exp = analyze.load_gene_tpms(rnaseq_file, top_n=2, known_ids=known)
        assert top_exp == {"ENSG00000001.1", "ENSG00000001",
                           "ENSG00000002.1", "ENSG00000002"}
        # Unranked genes keep their TPM.
        assert tpms["gSpikein_ERCC-00130"] == pytest.approx(900.0)


def test_load_gene_coords_top_expressed_and_nonexpressed():
    with tempfile.TemporaryDirectory() as tmpdir:
        gtf_file = os.path.join(tmpdir, "genes.gtf")
        entries = [
            # Expressed + strand (1-based GTF: 1000..2000 -> 0-based: 999..2000, TSS=999)
            {"gene_id": "ENSG00000001.1", "gene_name": "GENE1", "chrom": "chr1",
             "start": 1000, "end": 2000, "strand": "+"},
            # Expressed - strand (1-based GTF: 3000..4000 -> 0-based: 2999..4000, TSS=3999)
            {"gene_id": "ENSG00000002.5", "gene_name": "GENE2", "chrom": "chr1",
             "start": 3000, "end": 4000, "strand": "-"},
            # Non-expressed (TPM <= 0.1)
            {"gene_id": "ENSG00000003.1", "gene_name": "GENE3", "chrom": "chr2",
             "start": 5000, "end": 6000, "strand": "+"},
            # Intermediate TPM (not in top expressed, but TPM > 0.1: should be excluded from both)
            {"gene_id": "ENSG00000004.1", "gene_name": "GENE4", "chrom": "chr2",
             "start": 7000, "end": 8000, "strand": "+"},
        ]
        write_gtf(gtf_file, entries)

        gene_tpms = {
            "ENSG00000001.1": 50.0, "ENSG00000001": 50.0,
            "ENSG00000002.5": 20.0, "ENSG00000002": 20.0,
            "ENSG00000003.1": 0.05, "ENSG00000003": 0.05,
            "ENSG00000004.1": 5.0, "ENSG00000004": 5.0,
        }
        top_exp_ids = {"ENSG00000001.1", "ENSG00000001", "ENSG00000002.5", "ENSG00000002"}

        exp_b, exp_t, nonexp_b, nonexp_t = analyze.load_gene_coords(
            gtf_file, gene_tpms, top_expressed_ids=top_exp_ids, nonexp_thresh=0.1
        )

        assert len(exp_b) == 2
        assert len(exp_tss := exp_t) == 2
        assert len(nonexp_b) == 1
        assert len(nonexp_t) == 1

        # Check coordinate conversion
        assert exp_b[0] == ("chr1", 999, 2000, "GENE1")
        assert exp_tss[0] == ("chr1", 999, 1000, "GENE1")

        assert exp_b[1] == ("chr1", 2999, 4000, "GENE2")
        assert exp_tss[1] == ("chr1", 3999, 4000, "GENE2")

        assert nonexp_b[0] == ("chr2", 4999, 6000, "GENE3")
        assert nonexp_t[0] == ("chr2", 4999, 5000, "GENE3")


def test_load_gene_coords_version_independent_matching():
    with tempfile.TemporaryDirectory() as tmpdir:
        gtf_file = os.path.join(tmpdir, "genes.gtf")
        # GTF has unversioned ID, RNA-seq had versioned ID
        entries = [
            {"gene_id": "ENSG00000099", "gene_name": "GENE99", "chrom": "chr1",
             "start": 100, "end": 500, "strand": "+"},
        ]
        write_gtf(gtf_file, entries)

        # Top expressed has base and versioned IDs
        top_exp_ids = {"ENSG00000099.3", "ENSG00000099"}
        gene_tpms = {"ENSG00000099.3": 40.0, "ENSG00000099": 40.0}

        exp_b, exp_t, nonexp_b, nonexp_t = analyze.load_gene_coords(
            gtf_file, gene_tpms, top_expressed_ids=top_exp_ids
        )
        assert len(exp_b) == 1
        assert exp_b[0] == ("chr1", 99, 500, "GENE99")


def test_load_gene_coords_fallback_threshold():
    with tempfile.TemporaryDirectory() as tmpdir:
        gtf_file = os.path.join(tmpdir, "genes.gtf")
        entries = [
            {"gene_id": "ENSG00000001", "gene_name": "G1", "chrom": "chr1",
             "start": 100, "end": 500, "strand": "+"},
            {"gene_id": "ENSG00000002", "gene_name": "G2", "chrom": "chr1",
             "start": 600, "end": 900, "strand": "+"},
        ]
        write_gtf(gtf_file, entries)

        gene_tpms = {"ENSG00000001": 2.5, "ENSG00000002": 0.05}
        exp_b, exp_t, nonexp_b, nonexp_t = analyze.load_gene_coords(
            gtf_file, gene_tpms, top_expressed_ids=None, exp_thresh=1.0, nonexp_thresh=0.1
        )
        assert len(exp_b) == 1
        assert exp_b[0][3] == "G1"
        assert len(nonexp_b) == 1
        assert nonexp_b[0][3] == "G2"


def test_gene_name_matching():
    with tempfile.TemporaryDirectory() as tmpdir:
        gtf_file = os.path.join(tmpdir, "genes.gtf")
        entries = [
            {"gene_id": "NOVEL_ID", "gene_name": "KNOWN_NAME", "chrom": "chr1",
             "start": 100, "end": 500, "strand": "+"},
        ]
        write_gtf(gtf_file, entries)

        # Matched via gene_name
        top_exp_ids = {"KNOWN_NAME"}
        gene_tpms = {"KNOWN_NAME": 15.0}

        exp_b, exp_t, nonexp_b, nonexp_t = analyze.load_gene_coords(
            gtf_file, gene_tpms, top_expressed_ids=top_exp_ids
        )
        assert len(exp_b) == 1
        assert exp_b[0][3] == "KNOWN_NAME"


def test_make_expressed_annotations_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        rnaseq_file = os.path.join(tmpdir, "rnaseq.tsv")
        gtf_file = os.path.join(tmpdir, "genes.gtf.gz")

        rows = [
            {"gene_id": "ENSG00000001.1", "TPM": "80.0"},
            {"gene_id": "ENSG00000002.1", "TPM": "0.0"},
        ]
        write_rnaseq_tsv(rnaseq_file, rows)

        entries = [
            {"gene_id": "ENSG00000001.1", "gene_name": "ACTB", "chrom": "chr1",
             "start": 10000, "end": 20000, "strand": "+"},
            {"gene_id": "ENSG00000002.1", "gene_name": "SILENT", "chrom": "chr1",
             "start": 30000, "end": 40000, "strand": "-"},
        ]
        write_gtf(gtf_file, entries, gzipped=True)

        annotations = analyze.make_expressed_annotations(
            rnaseq_file, gtf_file, top_expressed=12000, nonexp_thresh=0.1
        )

        labels = [name for name, _ in annotations]
        assert "ExpressedGeneBodies" in labels
        assert "ExpressedTSS" in labels
        assert "ExpressedTSS2kb" in labels
        assert "NonExpressedGeneBodies" in labels
        assert "NonExpressedTSS" in labels
        assert "NonExpressedTSS2kb" in labels

        annot_dict = dict(annotations)
        # Check ExpressedTSS2kb window: TSS interval [9999, 10000) extended by 2000 bp -> [7999, 12000)
        assert annot_dict["ExpressedTSS2kb"][0] == ("chr1", 7999, 12000, "ACTB")
        # Check NonExpressedTSS2kb window: TSS interval [39999, 40000) extended by 2000 bp -> [37999, 42000)
        assert annot_dict["NonExpressedTSS2kb"][0] == ("chr1", 37999, 42000, "SILENT")


def test_state_predicates():
    assert utils.is_promoter_core("Tss")
    assert utils.is_promoter_core("1_TssA")
    assert utils.is_promoter_core("2_TssAFlnk")
    assert not utils.is_promoter_core("TssFlnkU")
    assert not utils.is_promoter_core("TssFlnk")
    assert not utils.is_promoter_core("TssFlnkD")

    assert utils.is_promoter_flank("TssFlnk")
    assert utils.is_promoter_flank("TssFlnkU")
    assert utils.is_promoter_flank("TssFlnkD")
    assert not utils.is_promoter_flank("Tss")

    assert utils.is_promoter("Tss")
    assert utils.is_promoter("1_TssA")
    assert utils.is_promoter("TssFlnk")
    assert not utils.is_promoter("Tx")
    assert not utils.is_promoter("Enh")

    assert utils.is_tx("Tx")
    assert utils.is_tx("2_TxA")
    assert utils.is_tx("TxWk")
    assert not utils.is_tx("Tss")

    assert utils.is_tx_core("Tx")
    assert utils.is_tx_core("TxA")
    assert not utils.is_tx_core("TxWk")
    assert not utils.is_tx_core("EnhG1")

    assert utils.is_enhancer("Enh")
    assert utils.is_enhancer("5_EnhA")
    assert utils.is_enhancer("EnhG1")
    assert not utils.is_enhancer("EnhBiv")
    assert not utils.is_enhancer("Tss")

    assert utils.is_distal_enhancer("Enh")
    assert utils.is_distal_enhancer("5_EnhA")
    assert utils.is_distal_enhancer("Enh1")
    assert not utils.is_distal_enhancer("EnhG1")

    assert utils.is_genic_enhancer("EnhG")
    assert utils.is_genic_enhancer("EnhG1")
    assert not utils.is_genic_enhancer("Enh1")

    assert utils.is_active("1_TssA")
    assert utils.is_active("EnhG")
    assert not utils.is_active("Tx")
    assert not utils.is_active("Quies")

    assert utils.is_active_open("1_TssA")
    assert utils.is_active_open("5_EnhA")
    assert utils.is_active_open("Enh1")
    assert not utils.is_active_open("EnhG1")
    assert not utils.is_active_open("TssFlnkD")

    assert utils.is_biv("TssBiv")
    assert utils.is_biv("10_Biv")
    assert utils.is_biv("EnhBiv")
    assert not utils.is_biv("Tss")


def test_compute_enrichment_50_percent_overlap():
    # Segments
    # Seg1: chr1 1000..1600 (1_TssA, 600bp)
    # Seg2: chr1 3000..3400 (1_TssA, 400bp)
    # Seg3: chr1 3400..3800 (2_TssAFlnk, 400bp)
    # Seg4: chr1 5000..6000 (1_TssA, 1000bp)
    # Seg5: chr1 7000..8000 (15_Quies, 1000bp)
    segs = [
        ("chr1", 1000, 1600, "1_TssA"),
        ("chr1", 3000, 3400, "1_TssA"),
        ("chr1", 3400, 3800, "2_TssAFlnk"),
        ("chr1", 5000, 6000, "1_TssA"),
        ("chr1", 7000, 8000, "15_Quies"),
    ]

    # Two annotation loci
    # Ann1: chr1 1000..2000 (1000bp)
    # Ann2: chr1 3000..4000 (1000bp)
    ann_segs = [
        ("chr1", 1000, 2000),
        ("chr1", 3000, 4000),
    ]

    df = analyze.compute_enrichment(segs, [("TestAnnotation", ann_segs)])
    assert "sensitivity_50" in df.columns
    assert "coverage_50" in df.columns

    by_state = df.set_index("state")

    # 1_TssA:
    # Locus 1: 600bp/1000bp = 60% >= 50% (hit)
    # Locus 2: 400bp/1000bp = 40% < 50% (miss)
    # sensitivity_50 = 1 / 2 = 0.5
    assert by_state.loc["1_TssA", "sensitivity_50"] == pytest.approx(0.5)

    # 1_TssA segments: Seg1 (100%), Seg2 (100%), Seg4 (0%) -> 2 hits out of 3 segs
    assert by_state.loc["1_TssA", "coverage_50"] == pytest.approx(2.0 / 3.0)

    # 2_TssAFlnk:
    # Locus 1: 0bp (miss), Locus 2: 400bp (40% < 50% -> miss) -> sensitivity_50 = 0.0
    # Seg3 (100%) -> coverage_50 = 1.0
    assert by_state.loc["2_TssAFlnk", "sensitivity_50"] == pytest.approx(0.0)
    assert by_state.loc["2_TssAFlnk", "coverage_50"] == pytest.approx(1.0)

    # POOL:Tss (1_TssA + 2_TssAFlnk):
    # Locus 1: 600bp = 60% >= 50% (hit)
    # Locus 2: 400bp + 400bp = 800bp = 80% >= 50% (hit)
    # sensitivity_50 = 2 / 2 = 1.0
    assert by_state.loc["POOL:Tss", "sensitivity_50"] == pytest.approx(1.0)
    # Family segments: Seg1, Seg2, Seg3, Seg4 (3 hits out of 4 segs) -> coverage_50 = 3/4 = 0.75
    assert by_state.loc["POOL:Tss", "coverage_50"] == pytest.approx(0.75)


def test_stacked_bar_plot_xticklabels():
    import pandas as pd
    df = pd.DataFrame({
        "State1": [0.3, 0.4],
        "State2": [0.7, 0.6],
    }, index=["kmeans_homer", "kmeans_macs2"])

    ax = utils.stacked_bar_plot(df, xticklabels="group")
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["Homer KMeans", "MACS2 KMeans"]


def test_annotation_sensitivity_precision():
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmpdir:
        enr_dir = os.path.join(tmpdir, "enrichment")
        os.makedirs(enr_dir)
        enr_df = pd.DataFrame({
            "label": ["atac_peaks", "atac_peaks"],
            "state": ["1_TssA", "2_TssAFlnk"],
            "fold_enrichment": [2.0, 2.0],
            "coverage": [0.1, 0.1],
        })
        enr_df.to_csv(os.path.join(enr_dir, "enrichment.tsv"), sep="\t", index=False)
        rep_df = pd.DataFrame({
            "state": ["1_TssA", "2_TssAFlnk"],
            "total_bp": [1000, 1000],
        })
        rep_df.to_csv(os.path.join(tmpdir, "report.tsv"), sep="\t", index=False)

        res = utils.annotation_sensitivity_precision(tmpdir, ("1_TssA",), "atac_")
        assert len(res) == 1
        assert utils.PRECISION_DISPLAY in res[0]
        assert utils.SENSITIVITY_DISPLAY in res[0]
        assert res[0]["Label"] == "atac_peaks"
        # Deprecated alias works identically
        res_alias = utils.annotation_sensitivity_specificity(tmpdir, ("1_TssA",), "atac_")
        assert res_alias == res


def test_plot_summary_no_legend():
    import pandas as pd
    import summary_plots

    df = pd.DataFrame({
        "ds1": [1.0, 2.0],
        "ds2": [1.5, 2.5]
    }, index=["kmeans_homer", "kmeans_macs2"])

    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, "test_summary.pdf")
        summary_plots._plot_summary(df, "Test Title", "Y Label", outpath)
        assert os.path.exists(outpath)


def test_global_bin_constants_and_helpers():
    assert utils.CHROMHMM_BIN == 200
    assert utils.PEAKS_BIN == 100

    assert utils.method_bin_size("chromhmm_default") == 200
    assert utils.method_bin_size("joint_chromhmm") == 200
    assert utils.method_bin_size("ref") == 200
    assert utils.method_bin_size("kmeans_homer") == 100
    assert utils.method_bin_size("kmeans_macs2") == 100
    assert utils.method_bin_size("kmeans_omni") == 100
    assert utils.method_bin_size("joint_kmeans_homer") == 100
    assert utils.method_bin_size("joint_kmeans_macs2") == 100
    assert utils.method_bin_size("joint_kmeans_omni") == 100

    assert utils.seg_bin_size("ds/chromhmm_default_result/cell_15_dense_matched.bed") == 200
    assert utils.seg_bin_size("ds/omni/omni_kmeans_states_matched.bed") == 100
    assert utils.seg_bin_size("ds/homer/homer_kmeans_states_matched.bed") == 100
    assert utils.seg_bin_size("ds/macs2/macs2_kmeans_states_matched.bed") == 100
    assert utils.seg_bin_size("ds/ENCFF123ABC.bed") == 200

    assert utils.get_bin_size("chromhmm_default") == 200
    assert utils.get_bin_size("kmeans_homer") == 100
    assert utils.get_bin_size("ds/omni/omni_kmeans_states_matched.bed") == 100


def test_compute_consistency_curve():
    # Synthetic counts dict for 2 states over 3 segmentations:
    # State A: depth cumulative <= 1: 10, <= 2: 15, <= 3: 20
    # State B: depth cumulative <= 1: 30, <= 2: 45, <= 3: 50
    counts = {
        "StateA": {1: 10, 2: 15, 3: 20},
        "StateB": {1: 30, 2: 45, 3: 50},
    }
    # Physical genome size G = (sum_s sum_{n=1}^3 shared_s(n)) / 3
    # StateA: (20-0) + (20-10) + (20-15) = 20 + 10 + 5 = 35
    # StateB: (50-0) + (50-30) + (50-45) = 50 + 20 + 5 = 75
    # Total sample bins = 35 + 75 = 110, G = 110 / 3
    # N=1 (shared >= 1): (20 - 0) + (50 - 0) = 70 -> frac = 70 / (110/3)
    # N=2 (shared >= 2): (20 - 10) + (50 - 30) = 10 + 20 = 30 -> frac = 30 / (110/3)
    # N=3 (shared >= 3): (20 - 15) + (50 - 45) = 5 + 5 = 10 -> frac = 10 / (110/3)
    assert analyze.compute_genome_size(counts) == pytest.approx(110.0 / 3.0)

    curve = analyze.compute_consistency_curve(counts)
    assert len(curve) == 3
    assert curve[0] == (1, 70, pytest.approx(70.0 / (110.0 / 3.0)))
    assert curve[1] == (2, 30, pytest.approx(30.0 / (110.0 / 3.0)))
    assert curve[2] == (3, 10, pytest.approx(10.0 / (110.0 / 3.0)))

    # Explicit genome_size = 70
    curve_g70 = analyze.compute_consistency_curve(counts, genome_size=70)
    assert curve_g70[0] == (1, 70, 1.0)
    assert curve_g70[1] == (2, 30, pytest.approx(30.0 / 70.0))
    assert curve_g70[2] == (3, 10, pytest.approx(10.0 / 70.0))

    # Test with exclude_quiescent=True
    counts_with_quies = {
        "1_TssA": {1: 10, 2: 15, 3: 20},
        "15_Quies": {1: 100, 2: 200, 3: 300},
    }
    # Full genome size G = (35 + (300 + 200 + 100)) / 3 = (35 + 600) / 3 = 635 / 3
    assert analyze.compute_genome_size(counts_with_quies) == pytest.approx(635.0 / 3.0)
    curve_no_quies = analyze.compute_consistency_curve(counts_with_quies, exclude_quiescent=True)
    assert len(curve_no_quies) == 3
    # Only 1_TssA bins normalized by full G:
    # N=1: 20 / (635/3) = 60/635
    # N=2: (20 - 10) / (635/3) = 30/635
    # N=3: (20 - 15) / (635/3) = 15/635
    assert curve_no_quies[0] == (1, 20, pytest.approx(60.0 / 635.0))
    assert curve_no_quies[1] == (2, 10, pytest.approx(30.0 / 635.0))
    assert curve_no_quies[2] == (3, 5, pytest.approx(15.0 / 635.0))

    # All quiescent states excluded
    assert analyze.compute_consistency_curve({"Quies": {1: 10, 2: 20}}, exclude_quiescent=True) == []

    # Empty counts
    assert analyze.compute_consistency_curve({}) == []
    assert analyze.compute_genome_size({}) == 0


def test_plot_method_consistency(tmp_path):
    counts = {
        "1_TssA": {1: 10, 2: 20},
        "2_Quies": {1: 5, 2: 30},
    }
    outpath = str(tmp_path / "method_consistency.pdf")
    analyze.plot_method_consistency(counts, "Test Model", outpath, ylabel="Fraction of genome")
    assert os.path.exists(outpath)

    outpath_no_quies = str(tmp_path / "method_consistency_no_quies.pdf")
    analyze.plot_method_consistency(counts, "Test Model", outpath_no_quies, exclude_quiescent=True)
    assert os.path.exists(outpath_no_quies)


def test_compute_stacked_consistency():
    counts = {
        "StateA": {1: 10, 2: 15, 3: 18, 4: 20},
        "StateB": {1: 30, 2: 45, 3: 50, 4: 60},
    }
    # M = 4
    # StateA: (20-0) + (20-10) + (20-15) + (20-18) = 20 + 10 + 5 + 2 = 37
    # StateB: (60-0) + (60-30) + (60-45) + (60-50) = 60 + 30 + 15 + 10 = 115
    # G = (37 + 115) / 4 = 152 / 4 = 38.0
    # 0%: n = 1 -> StateA: 20/38, StateB: 60/38, Total = 80/38
    # 25%: n = 1 -> StateA: 20/38, StateB: 60/38
    # 50%: n = 2 -> StateA: (20-10)/38 = 10/38, StateB: (60-30)/38 = 30/38
    # 75%: n = 3 -> StateA: (20-15)/38 = 5/38, StateB: (60-45)/38 = 15/38
    # 100%: n = 4 -> StateA: (20-18)/38 = 2/38, StateB: (60-50)/38 = 10/38
    df = analyze.compute_stacked_consistency(counts)
    expected_indices = ["0%", "2%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "98%", "100%"]
    assert list(df.index) == expected_indices
    assert list(df.columns) == ["StateA", "StateB"]
    assert df.loc["0%", "StateA"] == pytest.approx(20.0 / 38.0)
    assert df.loc["0%", "StateB"] == pytest.approx(60.0 / 38.0)
    assert df.loc["2%", "StateA"] == pytest.approx(20.0 / 38.0)
    assert df.loc["10%", "StateA"] == pytest.approx(20.0 / 38.0)
    assert df.loc["50%", "StateA"] == pytest.approx(10.0 / 38.0)
    assert df.loc["50%", "StateB"] == pytest.approx(30.0 / 38.0)
    assert df.loc["75%", "StateA"] == pytest.approx(5.0 / 38.0)
    assert df.loc["75%", "StateB"] == pytest.approx(15.0 / 38.0)
    assert df.loc["90%", "StateA"] == pytest.approx(2.0 / 38.0)
    assert df.loc["90%", "StateB"] == pytest.approx(10.0 / 38.0)
    assert df.loc["95%", "StateA"] == pytest.approx(2.0 / 38.0)
    assert df.loc["98%", "StateA"] == pytest.approx(2.0 / 38.0)
    assert df.loc["100%", "StateA"] == pytest.approx(2.0 / 38.0)
    assert df.loc["100%", "StateB"] == pytest.approx(10.0 / 38.0)

    # Test with custom fractions
    custom_df = analyze.compute_stacked_consistency(counts, fractions=(0.0, 0.5, 1.0))
    assert list(custom_df.index) == ["0%", "50%", "100%"]

    # Test with explicit genome_size
    df_g80 = analyze.compute_stacked_consistency(counts, genome_size=80)
    assert df_g80.loc["0%", "StateA"] == pytest.approx(20.0 / 80.0)
    assert df_g80.loc["0%", "StateB"] == pytest.approx(60.0 / 80.0)

    # Test with fine-grained M=100
    counts_100 = {
        "StateA": {i: i * 10 for i in range(1, 101)},
        "StateB": {i: i * 20 for i in range(1, 101)},
    }
    # For StateA, depths[i] = i * 10.
    # sum_{n=1}^100 (1000 - (n-1)*10) = 100*1000 - 10 * sum_{k=0}^99 k = 100000 - 10 * (99*100/2) = 100000 - 49500 = 50500
    # For StateB, sum = 2 * 50500 = 101000
    # Total sample bins = 50500 + 101000 = 151500
    # G = 151500 / 100 = 1515.0
    df_100 = analyze.compute_stacked_consistency(counts_100)
    assert list(df_100.index) == expected_indices
    # 2%: n = round(0.02 * 100) = 2 -> StateA: (1000 - 10) / 1515 = 990 / 1515
    # 10%: n = round(0.10 * 100) = 10 -> StateA: (1000 - 90) / 1515 = 910 / 1515
    # 75%: n = round(0.75 * 100) = 75 -> StateA: (1000 - 740) / 1515 = 260 / 1515
    # 90%: n = round(0.90 * 100) = 90 -> StateA: (1000 - 890) / 1515 = 110 / 1515
    # 95%: n = round(0.95 * 100) = 95 -> StateA: (1000 - 940) / 1515 = 60 / 1515
    # 98%: n = round(0.98 * 100) = 98 -> StateA: (1000 - 970) / 1515 = 30 / 1515
    # 100%: n = 100 -> StateA: (1000 - 990) / 1515 = 10 / 1515
    assert df_100.loc["2%", "StateA"] == pytest.approx(990.0 / 1515.0)
    assert df_100.loc["10%", "StateA"] == pytest.approx(910.0 / 1515.0)
    assert df_100.loc["75%", "StateA"] == pytest.approx(260.0 / 1515.0)
    assert df_100.loc["90%", "StateA"] == pytest.approx(110.0 / 1515.0)
    assert df_100.loc["95%", "StateA"] == pytest.approx(60.0 / 1515.0)
    assert df_100.loc["98%", "StateA"] == pytest.approx(30.0 / 1515.0)
    assert df_100.loc["100%", "StateA"] == pytest.approx(10.0 / 1515.0)

    # Empty counts
    assert analyze.compute_stacked_consistency({}).empty


def test_plot_stacked_consistency(tmp_path):
    counts = {
        "1_TssA": {1: 10, 2: 15, 3: 20, 4: 25},
        "2_Quies": {1: 30, 2: 45, 3: 60, 4: 75},
    }
    # With custom colors dict (hex and rgb string)
    colors = {"1_TssA": "#FF0000", "2_Quies": "255,255,255"}
    outpath = str(tmp_path / "stacked_consistency_custom.pdf")
    analyze.plot_stacked_consistency(counts, "Test Model", outpath, colors=colors, ylabel="Fraction of genome")
    assert os.path.exists(outpath)

    # With colors=None (falls back to canonical markup state colors)
    outpath_default = str(tmp_path / "stacked_consistency_default.pdf")
    analyze.plot_stacked_consistency(counts, "Test Model", outpath_default, colors=None)
    assert os.path.exists(outpath_default)


def test_method_linestyle_and_normalization():
    assert utils.method_linestyle("chromhmm_default") == "-"
    assert utils.method_linestyle("joint_chromhmm") == "--"
    assert utils.method_linestyle("kmeans_homer") == "-"
    assert utils.method_linestyle("joint_kmeans_homer") == "--"
    assert utils.method_linestyle("De-novo Homer KMeans") == "-"
    assert utils.method_linestyle("De-novo Joint Homer KMeans") == "--"
    assert utils.method_linestyle("15-state Default ChromHMM") == "-"
    assert utils.method_linestyle("15-state Joint ChromHMM") == "--"

    assert utils.normalize_method("15state_chromhmm_default") == "chromhmm_default"
    assert utils.normalize_method("15state_joint_chromhmm") == "joint_chromhmm"
    assert utils.normalize_method("denovo_kmeans_homer") == "kmeans_homer"
    assert utils.normalize_method("denovo_joint_kmeans_homer") == "joint_kmeans_homer"
    assert utils.normalize_method("denovo_kmeans_macs2") == "kmeans_macs2"
    assert utils.normalize_method("denovo_joint_kmeans_macs2") == "joint_kmeans_macs2"
    assert utils.normalize_method("denovo_kmeans_omni") == "kmeans_omni"
    assert utils.normalize_method("denovo_joint_kmeans_omni") == "joint_kmeans_omni"


def test_plot_methods_consistency(tmp_path):
    counts_a = {
        "1_TssA": {1: 10, 2: 20},
        "2_Quies": {1: 5, 2: 30},
    }
    counts_b = {
        "1_TssA": {1: 8, 2: 20},
        "2_Quies": {1: 12, 2: 30},
    }

    # Test with list of tuples
    models = [
        ("15-state Default ChromHMM", counts_a, "chromhmm_default"),
        ("15-state Joint ChromHMM", counts_b, "joint_chromhmm"),
        ("De-novo Homer KMeans", counts_a, "kmeans_homer"),
        ("De-novo Joint Homer KMeans", counts_b, "joint_kmeans_homer"),
    ]
    outpath_list = str(tmp_path / "consistency_methods_list.pdf")
    analyze.plot_methods_consistency(models, "Method consistency", outpath_list)
    assert os.path.exists(outpath_list)

    # Test with DataFrame
    df_list = []
    for title, counts, m_key in models:
        curve = analyze.compute_consistency_curve(counts)
        df_m = pd.DataFrame(curve, columns=["N", "SharedBins", "Fraction"])
        df_m["Model"] = title
        df_m["Slug"] = m_key
        df_list.append(df_m)
    df_all = pd.concat(df_list, ignore_index=True)

    outpath_df = str(tmp_path / "consistency_methods_df.pdf")
    analyze.plot_methods_consistency(df_all, "Method consistency", outpath_df)
    assert os.path.exists(outpath_df)

    # Test with dict
    dict_data = {
        "Default ChromHMM": counts_a,
        "Joint ChromHMM": counts_b,
    }
    outpath_dict = str(tmp_path / "consistency_methods_dict.pdf")
    analyze.plot_methods_consistency(dict_data, "Method consistency", outpath_dict, ylabel="Fraction of genome")
    assert os.path.exists(outpath_dict)

    # Test with custom styles
    custom_styles = {
        "Default ChromHMM": {"color": "#FF0000", "linestyle": ":"},
        "Joint ChromHMM": {"color": "#00FF00", "linestyle": "-."},
    }
    outpath_styles = str(tmp_path / "consistency_methods_styles.pdf")
    analyze.plot_methods_consistency(dict_data, "Method consistency", outpath_styles, styles=custom_styles)
    assert os.path.exists(outpath_styles)

    # Test with exclude_quiescent=True
    outpath_no_quies = str(tmp_path / "consistency_methods_no_quies.pdf")
    analyze.plot_methods_consistency(dict_data, "Method consistency", outpath_no_quies, exclude_quiescent=True)
    assert os.path.exists(outpath_no_quies)

    # Test with yscale="log1p"
    outpath_log1p = str(tmp_path / "consistency_methods_log1p.pdf")
    analyze.plot_methods_consistency(df_all, "Method consistency", outpath_log1p, yscale="log1p")
    assert os.path.exists(outpath_log1p)

    outpath_method_log1p = str(tmp_path / "method_consistency_log1p.pdf")
    analyze.plot_method_consistency(counts_a, "Method consistency", outpath_method_log1p, yscale="log1p")
    assert os.path.exists(outpath_method_log1p)

    # Test discrete x-axis consistency
    fractions = (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 1.0)
    curve_disc = analyze.compute_consistency_curve(counts_a, fractions=fractions)
    assert len(curve_disc) == len(fractions)
    assert curve_disc[0][0] == "0%"
    assert curve_disc[-1][0] == "100%"

    # Test discrete x-axis consistency starting with 25% (e.g. for non-quiescent consistency)
    fractions_nq = (0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 1.0)
    curve_disc_nq = analyze.compute_consistency_curve(counts_a, fractions=fractions_nq, exclude_quiescent=True)
    assert len(curve_disc_nq) == len(fractions_nq)
    assert curve_disc_nq[0][0] == "25%"
    assert curve_disc_nq[-1][0] == "100%"

    # Test quiescent fraction consistency
    # counts_a: 1_TssA: {1: 10, 2: 20}, 2_Quies: {1: 5, 2: 30}
    # n=1 (0%): shared_all = (20-0) + (30-0) = 50, shared_quies = 30 -> ratio = 30/50 = 0.6
    # n=2 (100%): shared_all = (20-10) + (30-5) = 10 + 25 = 35, shared_quies = 25 -> ratio = 25/35 = 5/7 ≈ 0.7143
    curve_q = analyze.compute_quiescent_consistency_curve(counts_a, fractions=fractions)
    assert len(curve_q) == len(fractions)
    assert curve_q[0][0] == "0%"
    assert curve_q[0][2] == pytest.approx(30.0 / 50.0)
    assert curve_q[-1][0] == "100%"
    assert curve_q[-1][2] == pytest.approx(25.0 / 35.0)

    outpath_method_q = str(tmp_path / "method_consistency_quies.pdf")
    analyze.plot_method_consistency(counts_a, "Method consistency Quies", outpath_method_q, quiescent_fraction=True, fractions=fractions)
    assert os.path.exists(outpath_method_q)

    outpath_methods_q = str(tmp_path / "consistency_methods_quies.pdf")
    analyze.plot_methods_consistency(dict_data, "Quiescent fraction of shared genome", outpath_methods_q, quiescent_fraction=True, fractions=fractions)
    assert os.path.exists(outpath_methods_q)

    df_disc_list = []
    for title, counts, m_key in models:
        curve = analyze.compute_consistency_curve(counts, fractions=fractions)
        df_m = pd.DataFrame(curve, columns=["Percent", "SharedBins", "Fraction"])
        df_m["Model"] = title
        df_m["Slug"] = m_key
        df_disc_list.append(df_m)
    df_disc_all = pd.concat(df_disc_list, ignore_index=True)

    outpath_disc_df = str(tmp_path / "consistency_methods_discrete_df.pdf")
    analyze.plot_methods_consistency(df_disc_all, "Method consistency (Discrete)", outpath_disc_df, yscale="log1p")
    assert os.path.exists(outpath_disc_df)

    # Test dataset-specific consistency plots
    df_disc_all["Dataset"] = "encode"
    outpath_ds = str(tmp_path / "consistency_methods_encode.pdf")
    analyze.plot_methods_consistency(df_disc_all[df_disc_all["Dataset"] == "encode"], "Method consistency — ENCODE", outpath_ds, yscale="log1p")
    assert os.path.exists(outpath_ds)

    outpath_disc_dict = str(tmp_path / "consistency_methods_discrete_dict.pdf")
    analyze.plot_methods_consistency(dict_data, "Method consistency (Discrete)", outpath_disc_dict, fractions=fractions, yscale="log1p", figsize=(8, 4))
    assert os.path.exists(outpath_disc_dict)

    # Empty inputs
    analyze.plot_methods_consistency([], "Title", str(tmp_path / "empty.pdf"))
    analyze.plot_methods_consistency({}, "Title", str(tmp_path / "empty.pdf"))
    analyze.plot_methods_consistency(pd.DataFrame(), "Title", str(tmp_path / "empty.pdf"))


def test_distance_tss_discordance(tmp_path):
    """Test Distance-to-TSS tier computation, bed loading, discordance rate, and grid plotting."""
    # 1. Create a dummy GTF file
    gtf_file = tmp_path / "test.gtf"
    gtf_lines = [
        "# Dummy GTF",
        "chr1\tHAVANA\tgene\t1001\t2000\t.\t+\t.\tgene_id \"G1\";",  # TSS = 1000
        "chr1\tHAVANA\tgene\t10001\t12000\t.\t-\t.\tgene_id \"G2\";", # TSS = 11999
        "chr2\tHAVANA\tgene\t501\t1500\t.\t+\t.\tgene_id \"G3\";",   # TSS = 500
    ]
    gtf_file.write_text("\n".join(gtf_lines))

    chrom_sizes = {"chr1": 20000, "chr2": 10000}
    dist_tiers = analyze.build_genome_tss_distance_bins(str(gtf_file), chrom_sizes, bin_size=200)

    assert "chr1" in dist_tiers
    assert "chr2" in dist_tiers
    assert len(dist_tiers["chr1"]) == 100
    assert len(dist_tiers["chr2"]) == 50

    # chr1 bin 5 (midpoint = 5 * 200 + 100 = 1100): distance to TSS (1000) = 100 bp -> tier 0 ("0-500bp")
    assert dist_tiers["chr1"][5] == 0

    # 2. Create dummy BED files
    bed_a_file = tmp_path / "test_a.bed"
    bed_a_lines = [
        "chr1\t0\t2000\t1_TssA",
        "chr1\t2000\t10000\t7_Enh",
        "chr1\t10000\t20000\t15_Quies",
    ]
    bed_a_file.write_text("\n".join(bed_a_lines))

    bed_b_file = tmp_path / "test_b.bed"
    bed_b_lines = [
        "chr1\t0\t2000\t1_TssA",      # Matches bed_a
        "chr1\t2000\t10000\t8_Tx",       # Mismatches bed_a
        "chr1\t10000\t20000\t15_Quies", # Matches bed_a
    ]
    bed_b_file.write_text("\n".join(bed_b_lines))

    codes_a = analyze.load_bed_codes(str(bed_a_file), chrom_sizes, bin_size=200)
    codes_b = analyze.load_bed_codes(str(bed_b_file), chrom_sizes, bin_size=200)

    assert "chr1" in codes_a
    assert codes_a["chr1"][0] == analyze.STATE_CANONICAL_IDX["TssA"]

    discordance = analyze.compute_distance_tss_discordance(codes_a, codes_b, dist_tiers)
    assert len(discordance) == len(analyze.DISTANCE_TSS_LABELS)
    assert 0.0 <= discordance[0] <= 1.0

    # 3. Test grid plot with dummy data
    rows = []
    methods = [
        utils.CHROMHMM_DEFAULT, utils.KMEANS_HOMER, utils.KMEANS_MACS2, utils.KMEANS_OMNI,
        utils.JOINT_CHROMHMM, utils.JOINT_KMEANS_HOMER, utils.JOINT_KMEANS_MACS2, utils.JOINT_KMEANS_OMNI
    ]
    for m in methods:
        for t in ["Between", "Replicate"]:
            for lbl in analyze.DISTANCE_TSS_LABELS:
                rows.append({"Method": m, "Type": t, "Distance": lbl, "Discordance": 0.25})

    df_test = pd.DataFrame(rows)
    outpath_grid = str(tmp_path / "distance_tss_grid.pdf")
    analyze.plot_distance_tss_discordance_grid(df_test, out_path=outpath_grid)
    assert os.path.exists(outpath_grid)

    outpath_grid_4x1 = str(tmp_path / "distance_tss_grid_4x1.pdf")
    analyze.plot_distance_tss_discordance_grid(df_test, layout=(4, 1), figsize=(8, 10), out_path=outpath_grid_4x1)
    assert os.path.exists(outpath_grid_4x1)

    outpath_grid_2x2 = str(tmp_path / "distance_tss_grid_2x2.pdf")
    analyze.plot_distance_tss_discordance_grid(df_test, layout=(2, 2), figsize=(10, 7), out_path=outpath_grid_2x2)
    assert os.path.exists(outpath_grid_2x2)
