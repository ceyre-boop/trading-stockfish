"""
Test-only flags to allow legacy paths in controlled harnesses without weakening
canonical/official enforcement. These defaults are False and should only be
set to True inside specific test modules that require legacy compatibility.
"""

# When True, allows legacy evaluator paths in tests even if canonical env vars
# are present. Do not enable in production/offical modes.
CANONICAL_TEST_BYPASS: bool = False
