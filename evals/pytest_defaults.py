"""Set evaluation defaults before pytest loads third-party plugins."""

# This lives with the evals because the default is specific to DeepEval, even
# though pyproject.toml loads it for every pytest run.

import os

from evals.deepeval_defaults import configure_deepeval

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
configure_deepeval()
