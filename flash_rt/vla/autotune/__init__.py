"""CPU-only persistent campaign records and controller utilities.

Workers produce result files; only the controller owns a campaign ledger.
Importing this package does not load a model or submit a job.
"""

from .schema import Candidate, decode_backend, encode_backend, validate_result

__all__ = ['Candidate', 'decode_backend', 'encode_backend', 'validate_result']
