# lits_benchmark: Benchmark implementations for LiTS framework.
#
# Each benchmark module registers its datasets, transitions, and resources
# via decorators (@register_dataset, @register_transition, @register_resource).
# Import the module to trigger registration, e.g.:
#   import lits_benchmark.mapeval
#   import lits_benchmark.blocksworld
