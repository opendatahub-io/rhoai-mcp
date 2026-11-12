"""Set evaluation defaults before pytest loads third-party plugins."""

# This lives with the evals because the default is specific to DeepEval, even
# though pyproject.toml loads it for every pytest run.

import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
