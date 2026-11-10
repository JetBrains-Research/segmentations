#!/bin/bash
# Project root directory
ROOT=$(cd "$(dirname "$0")" && pwd)
source "$ROOT/match.sh"

# Joint segmentations across replicates of the ENCODE datasets.
# Please ensure that snakemake part was already processed, see README.md
DIR=~/data/2026_segmentations/encode
mkdir -p "$DIR"

CHROMSIZES="$DIR/hg38.chrom.sizes"
CHROMHMM_JAR="$DIR/ChromHMM/ChromHMM.jar"
GENOME=hg38
STATES=15
CHROMHMM_BIN=200
PEAKS_BIN=100
# Marks list for the ENCODE datasets
MARKS="H3K4me3,H3K27ac,H3K4me1,H3K36me3,H3K9me3,H3K27me3"
REPS="rep1,rep2"
# Datasets with two replicates, see config_encode.yaml

# ENCODE reference segmentation of a dataset
ref_bed() {
 case "$1" in
  imr90) echo "ENCFF714POQ_chromhmm.bed" ;;
  monocytes) echo "ENCFF227EMB_chromhmm.bed" ;;
  spleen) echo "ENCFF183AWD_chromhmm.bed" ;;
 esac
}

# Cell type name of a dataset
cell_name() {
 case "$1" in
  imr90) echo "IMR90" ;;
  monocytes) echo "Monocyte" ;;
  spleen) echo "Spleen" ;;
 esac
}


# 1. Joint KMeans replicates states processing
for ds in imr90 monocytes spleen
do
 echo "===================="; echo "$ds"
 cd "$DIR/$ds" || exit
 for PC in homer macs2 omni
 do
  echo "~~~~~~~~~~~~~~~~~~~~"; echo "$PC"
  echo "Collecting peaks"
  ALL_PEAKS=""
  for R in rep1 rep2
  do
   for M in H3K4me3 H3K27ac H3K4me1 H3K36me3 H3K9me3 H3K27me3
   do
    case "$PC" in
     omni)  P="$R/omni/${M}.peak" ;;
     homer) P="$R/homer/${M}.bed" ;;
     macs2) P="$R/macs2/${M}.bed" ;;
    esac
    if [[ ! -f "$P" ]]; then echo "Missing $ds/$P"; P=NONE; fi
    ALL_PEAKS="$ALL_PEAKS $P"
   done
  done
  echo "Joint learning"
  if [[ -z $(ls -A "joint_kmeans/$PC") ]]; then
    mkdir -p "joint_kmeans/$PC"
    python "$ROOT/scripts/rules/joint_peaks_segmentation.py" \
   --bin "$PEAKS_BIN" --chromsizes "$CHROMSIZES" --marks "$MARKS" --cells "$REPS" \
   --peaks $ALL_PEAKS --states "$STATES" --outdir "joint_kmeans/$PC";
  fi
 done
done

# 2. Match joint KMeans to the ENCODE reference
for ds in imr90 monocytes spleen
do
 echo "===================="; echo "$ds"
 cd "$DIR/$ds" || exit
 for PC in homer macs2 omni
 do
  echo "Matching $ds $PC joint KMeans to the ENCODE reference"
  match_joint "$(ref_bed "$ds")" "joint_kmeans/$PC/rep1_kmeans_joint_states.bed" \
  "joint_kmeans/$PC/rep2_kmeans_joint_states.bed" "$PC" "$PEAKS_BIN"
 done
done

# 3. Joint ChromHMM replicates states processing
for ds in imr90 monocytes spleen
do
 echo "===================="; echo "$ds"
 cd "$DIR/$ds" || exit
 echo "~~~~~~~~~~~~~~~~~~~~"; echo "Joint ChromHMM"
 mkdir -p joint_chromhmm
 JOINT_BINARIZED=$(mktemp -d)
 python "$ROOT/scripts/joint_chromhmm.py" concat \
 --rep1 rep1/chromhmm_default --rep2 rep2/chromhmm_default --outdir "$JOINT_BINARIZED"
 java -mx4000M -jar "$CHROMHMM_JAR" LearnModel \
 -p 8 -b "$CHROMHMM_BIN" "$JOINT_BINARIZED" joint_chromhmm "$STATES" "$GENOME"
 rm -rf "$JOINT_BINARIZED"
done

# 4. Match joint ChromHMM to the ENCODE reference
for ds in imr90 monocytes spleen
do
 echo "===================="; echo "$ds"
 cd "$DIR/$ds" || exit
 echo "Matching $ds joint ChromHMM to the ENCODE reference"
 match_joint "$(ref_bed "$ds")" "joint_chromhmm/rep1_${STATES}_dense.bed" \
 "joint_chromhmm/rep2_${STATES}_dense.bed" "chromhmm" "$CHROMHMM_BIN"
done

# 5. ChromHMM over peaks processing
for ds in imr90 monocytes spleen
do
 echo "===================="; echo "$ds"
 cd "$DIR/$ds" || exit
 CELL=$(cell_name "$ds")
 for PC in homer macs2 omni
 do
  echo "~~~~~~~~~~~~~~~~~~~~"; echo "ChromHMM over peaks: $PC"
  TMP_DIR=$(mktemp -d)
  TABLE="$TMP_DIR/table.txt"
  CS="$TMP_DIR/chromsizes.txt"
  grep -v "_" "$CHROMSIZES" > "$CS"
  > "$TABLE"
  for M in H3K4me3 H3K27ac H3K4me1 H3K36me3 H3K9me3 H3K27me3
  do
   case "$PC" in
    omni)  P="${M}.peak" ;;
    homer) P="${M}.bed" ;;
    macs2) P="${M}.bed" ;;
   esac
   if [[ -f "$PC/$P" ]]; then
     echo -e "${CELL}\t${M}\t${P}" >> "$TABLE"
   fi
  done

  java -mx4000M -jar "$CHROMHMM_JAR" BinarizeBed -peaks -b "$PEAKS_BIN" "$CS" "$PC" "$TABLE" "$TMP_DIR/binary"
  java -mx4000M -jar "$CHROMHMM_JAR" LearnModel -p 8 -b "$PEAKS_BIN" "$TMP_DIR/binary" "$TMP_DIR/model" "$STATES" "$GENOME"

  cp "$TMP_DIR/model/${CELL}_${STATES}_dense.bed" "$PC/${PC}_chromhmm_states.bed" 2>/dev/null || \
  cp "$TMP_DIR/model/${STATES}_dense.bed" "$PC/${PC}_chromhmm_states.bed"

  rm -rf "$TMP_DIR"
 done
done

# 6. Match ChromHMM over peaks to the ENCODE reference
for ds in imr90 monocytes spleen
do
 echo "===================="; echo "$ds"
 cd "$DIR/$ds" || exit
 for PC in homer macs2 omni
 do
  echo "Matching $ds $PC ChromHMM over peaks to the ENCODE reference"
  match_states "$(ref_bed "$ds")" "$PC/${PC}_chromhmm_states.bed" "$PC" "$PEAKS_BIN" "."
 done
done
