"""Central configuration. Every experiment records which settings produced it."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"

BENIGN_LABEL = "BENIGN"
ATTEMPTED_SUFFIX = "- Attempted"

# How to treat "X - Attempted" flows: payload-reliant attacks where no payload
# was actually delivered. "exclude" keeps FDP measuring what we claim it
# measures. "attack" and "benign" are run later as a sensitivity check.
ATTEMPTED_POLICY = "exclude"          # "exclude" | "attack" | "benign"

CALIBRATION_FILE = "Monday-WorkingHours.csv"
TEST_FILE = "Tuesday-WorkingHours.csv"

# Identity/timing columns: excluded from features so the detector cannot key
# on IP addresses or capture order. Retained in the cache as metadata (with a
# leading underscore) because host-level aggregation needs them.
ID_COLUMNS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"]

# Kept in the parquet cache as "_"-prefixed metadata: never model features,
# but required for aggregation, auditing and analyst context.
METADATA_COLUMNS = ["Src IP", "Dst IP", "Dst Port", "Protocol"]

# Dst Port is genuine observable behaviour, but including it lets the detector
# shortcut to "port 21 = suspicious". Excluded from the headline run; included
# later as an ablation.
EXCLUDE_FROM_FEATURES = ID_COLUMNS + ["Dst Port"]

# Fraction of the calibration day used to FIT the detector. The remainder
# produces the conformal null. Split conformal requires these to be disjoint,
# otherwise calibration scores are in-sample and exchangeability breaks.
TRAIN_FRACTION = 0.5

# Batch size for BH. Conformal p-values floor at 1/(n_cal+1); testing ~360k
# hypotheses in one family puts the k=1 BH threshold below that floor, so
# nothing can ever be rejected. Hourly batching is both the operational unit
# a SOC works in and the statistically workable one.
BATCH = "5min"

DEFAULT_Q = 0.05
RANDOM_SEED = 0

# Include the missingness indicator as a model feature. Excluded from the
# headline run: single-packet flows lacking inter-arrival times are arguably an
# artefact of flow construction rather than attacker behaviour. Set True to run
# the ablation.
INCLUDE_IMPUTED_FLAG = False

# Batches smaller than this cannot support a meaningful multiple-testing
# correction and are reported separately rather than silently yielding zero.
MIN_BATCH = 1000


# --- experiment bookkeeping -------------------------------------------------
# Results are written to figures/<EXPERIMENT>/ so negative and superseded runs
# survive rather than being overwritten by the next iteration.
EXPERIMENT = "exp01_flow_rarity"

# --- channel B: host-hour aggregation ---------------------------------------
# Unit of aggregation. "src" scores each source host; "pair" scores each
# (source, destination) pair, which catches targeted attacks but multiplies the
# hypothesis count.
AGGREGATION_UNIT = "src"          # "src" | "pair"

# A host-hour is labelled malicious if at least this fraction of its flows are
# attacks. 0.0 means "any attack flow at all", which is the strictest reading
# and makes the task harder: a host with one attack flow among thousands of
# benign ones still counts as malicious.
HOST_HOUR_ATTACK_MIN_FRAC = 0.0

# Host-hours with fewer flows than this are too sparse to feature-engineer.
MIN_FLOWS_PER_HOST_HOUR = 3
