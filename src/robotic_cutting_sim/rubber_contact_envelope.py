"""Rubber contact friction envelope (ADR 0023 §23.2 decision 4; task ENV1, plan §2.4).

An envelope file (schema ``rcs_rubber_contact_envelope_v1``; ``configs/rubber/contact_envelope_v*.json``) maps each
(contact law, cells through the thickness n_t) to the friction coefficients that formally scored contact protocols
support: ``sliding_mu_max`` and ``static_mu_max``. Only formally scored PASS rows establish a row; DIAGNOSTIC evidence
may only narrow it. A pair without an ESTABLISHED row has no envelope.

    env = load_envelope()                               # contact_envelope_v1 (or a later version by path)
    env.row("grid_node_penalty_coulomb_v3", 4)          # the row dict
    env.friction_range("grid_node_penalty_coulomb_v3", [4, 5, 6], lower=0.3)   # P2-6 range or raises
    env.check_friction("grid_node_penalty_coulomb_v3", 4, 0.7)                 # raises above the row

Refusals raise :class:`FrictionOutOfEnvelope` (``code = UNSUPPORTED_FRICTION_OUT_OF_ENVELOPE``): mu >= 1.0 for every
row (``contact_coulomb_ill_posed``), values above the row's limit, rows that are missing or NOT_ESTABLISHED. The file
is data: this module never widens a row. Material status: ``SIMULATOR_ONLY_UNCALIBRATED``.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "rcs_rubber_contact_envelope_v1"
REFUSAL_CODE = "UNSUPPORTED_FRICTION_OUT_OF_ENVELOPE"
ABSOLUTE_MU_LIMIT = 1.0                  # mu >= 1.0 refused everywhere (contact_coulomb_ill_posed)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENVELOPE = ROOT/"configs"/"rubber"/"contact_envelope_v1.json"
ROW_STATUSES = ("ESTABLISHED", "NOT_ESTABLISHED")
_ROW_KEYS_ESTABLISHED = ("law", "n_t", "status", "sliding_mu_max", "static_mu_max", "scored", "run_id", "protocol",
                         "protocol_sha256", "evidence")


class FrictionOutOfEnvelope(ValueError):
    """A friction value (or a (law, n_t) pair) outside the declared envelope; ``code`` is the refusal code."""
    code = REFUSAL_CODE

    def __init__(self, detail):
        self.detail = str(detail)
        super().__init__(f"{REFUSAL_CODE}: {detail}")


@dataclass(frozen=True)
class ContactEnvelope:
    id: str
    path: str
    sha256: str
    rows: tuple
    raw: dict

    def row(self, law, n_t):
        """The row dict of (law, n_t); raises FrictionOutOfEnvelope if the pair is not listed."""
        for r in self.rows:
            if r["law"] == law and int(r["n_t"]) == int(n_t):
                return r
        raise FrictionOutOfEnvelope(f"no envelope row for law {law!r} at n_t {n_t} ({self.id})")

    def established(self, law, n_t):
        r = self.row(law, n_t)
        if r["status"] != "ESTABLISHED":
            raise FrictionOutOfEnvelope(f"law {law!r} at n_t {n_t}: {r['status']} ({r.get('reason', '')}) "
                                        f"in {self.id}")
        return r

    def sliding_mu_max(self, law, n_ts):
        """min over the n_t set of the ESTABLISHED sliding limit (raises if any row is missing / not established)."""
        return min(float(self.established(law, n)["sliding_mu_max"]) for n in sorted(set(int(v) for v in n_ts)))

    def friction_range(self, law, n_ts, *, lower=0.3):
        """[lower, min_n_t sliding_mu_max] (P2-6 rule, plan §2.4); raises if empty or a row is not established."""
        hi = self.sliding_mu_max(law, n_ts)
        lo = float(lower)
        if not (math.isfinite(lo) and 0.0 < lo <= hi < ABSOLUTE_MU_LIMIT):
            raise FrictionOutOfEnvelope(f"empty friction range [{lo}, {hi}] for law {law!r} at n_t {sorted(n_ts)}")
        return lo, hi

    def check_friction(self, law, n_t, mu, *, kind="sliding"):
        """Raise unless 0 <= mu <= the row's ``sliding`` (default) or ``static`` limit and mu < 1.0."""
        mu = float(mu)
        if not math.isfinite(mu) or mu < 0.0:
            raise FrictionOutOfEnvelope(f"friction {mu!r} is not a finite non-negative number")
        if mu >= ABSOLUTE_MU_LIMIT:
            raise FrictionOutOfEnvelope(f"friction {mu} >= {ABSOLUTE_MU_LIMIT} (contact_coulomb_ill_posed)")
        key = {"sliding": "sliding_mu_max", "static": "static_mu_max"}[kind]
        limit = float(self.established(law, n_t)[key])
        if mu > limit:
            raise FrictionOutOfEnvelope(f"friction {mu} > {key} {limit} of law {law!r} at n_t {n_t} ({self.id})")
        return limit

    def receipt_row(self, law, n_t):
        r = self.row(law, n_t)
        return {"envelope": self.id, "envelope_sha256": self.sha256, "law": law, "n_t": int(n_t),
                "status": r["status"], "sliding_mu_max": r.get("sliding_mu_max"),
                "static_mu_max": r.get("static_mu_max"), "run_id": r.get("run_id"),
                "protocol_sha256": r.get("protocol_sha256")}


def parse_envelope(obj, *, path="<memory>", sha256=None):
    """Validate an envelope dict (schema, row keys, status, limits in (0, 1), static >= sliding not required)."""
    if obj.get("schema") != SCHEMA:
        raise ValueError(f"envelope schema must be {SCHEMA!r}")
    if not isinstance(obj.get("id"), str) or not obj["id"]:
        raise ValueError("envelope id must be a non-empty string")
    rows, seen = [], set()
    for i, r in enumerate(obj.get("rows", ())):
        if r.get("status") not in ROW_STATUSES:
            raise ValueError(f"row {i}: status {r.get('status')!r} not in {ROW_STATUSES}")
        key = (r.get("law"), r.get("n_t"))
        if not isinstance(key[0], str) or not isinstance(key[1], int) or key[1] < 1:
            raise ValueError(f"row {i}: law / n_t")
        if key in seen:
            raise ValueError(f"row {i}: duplicate (law, n_t) {key}")
        seen.add(key)
        if r["status"] == "ESTABLISHED":
            missing = [k for k in _ROW_KEYS_ESTABLISHED if k not in r]
            if missing:
                raise ValueError(f"row {i}: ESTABLISHED row missing {missing}")
            for k in ("sliding_mu_max", "static_mu_max"):
                v = r[k]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 < v < ABSOLUTE_MU_LIMIT:
                    raise ValueError(f"row {i}: {k} {v!r} not in (0, {ABSOLUTE_MU_LIMIT})")
            if len(r["protocol_sha256"]) != 64:
                raise ValueError(f"row {i}: protocol_sha256")
        elif r.get("sliding_mu_max") is not None or r.get("static_mu_max") is not None:
            raise ValueError(f"row {i}: a NOT_ESTABLISHED row has no limits")
        rows.append(dict(r))
    return ContactEnvelope(id=obj["id"], path=str(path), sha256=sha256, rows=tuple(rows), raw=obj)


def load_envelope(path=DEFAULT_ENVELOPE, *, expected_sha256=None):
    """Load and validate an envelope file; ``expected_sha256`` (a pin) is checked when given."""
    data = Path(path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError(f"envelope {path} sha256 {sha} != pinned {expected_sha256}")
    return parse_envelope(json.loads(data), path=path, sha256=sha)


__all__ = ["ContactEnvelope", "FrictionOutOfEnvelope", "REFUSAL_CODE", "ABSOLUTE_MU_LIMIT", "DEFAULT_ENVELOPE",
           "load_envelope", "parse_envelope"]
