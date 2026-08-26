"""Sequential check for the public root-level model factory export."""

from lits import get_lm
from lits.lm import get_lm as get_lm_from_lm


print("root get_lm export matches lits.lm:", get_lm is get_lm_from_lm)
