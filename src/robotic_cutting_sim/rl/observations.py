"""Batched observations for the vector env (ADR 0021 §8): deployable ``state`` and opt-in ``privileged_state``.

``obs["policy"]`` (both profiles) is the deployable ``state`` block, fixed order and dims:

=========================  ====  =====================================================================
term                       dim   meaning
=========================  ====  =====================================================================
commanded_edge_pose_world  7     commanded knife-edge pose XYZ (m) + unit XYZW, canonical w >= 0
                                 (equals the measured pose under ideal tracking, workload A)
time_remaining_s           1     (max_steps - step_count) * control_dt
previous_action            8     executed normalized ``edge_delta_v1`` action of the previous step
                                 (HOLD row ``[0]*6 + [-1, 0]`` after a rejection; zeros after reset)
arm_flag                   1     1.0 if the previous executed action armed the cut, else 0.0
step_fraction              1     step_count / max_steps
=========================  ====  =====================================================================

It never contains material state, cut codes, components, contact truth, wrenches, forces or
generator truth (``policy_env.PRIVILEGED_KEYS``). :func:`assert_policy_terms_deployable` enforces
that by name at construction, and :meth:`StateObservation.compute` refuses privileged inputs under
the ``state`` profile. Consequence (ADR §8): TRIM hides the bump count, so a state-only policy is
blind to the geometry; geometry-aware deployable observations need a camera/height-map capability.

``obs["critic"]`` exists only with ``observation_profile="privileged_state"`` (explicit opt-in):
the policy block followed by privileged oracle terms. Privileged values never enter ``"policy"``.

Term specs and the guard are torch-free; Torch is imported only to assemble tensors.
"""
import re

from ..policy_env import PRIVILEGED_KEYS

OBSERVATION_VERSION = "rl_state_obs_v1"
PROFILES = ("state", "privileged_state")
POLICY_TERMS = (
    ("commanded_edge_pose_world", 7, "m, XYZW", "commanded knife-edge pose (canonical w >= 0)"),
    ("time_remaining_s", 1, "s", "(max_steps - step_count) * control_dt"),
    ("previous_action", 8, "normalized", "executed edge_delta_v1 action of the previous step"),
    ("arm_flag", 1, "0/1", "previous executed action armed the cut"),
    ("step_fraction", 1, "1", "step_count / max_steps"),
)
DEFAULT_CRITIC_TERMS = (
    ("target_bond_progress_fraction", 1, "1", "trim_bond_progress_packed_v1 severed target-bond fraction (oracle)"),
    ("off_target_bond_fraction", 1, "1", "severed substrate bonds / target bonds (oracle)"),
    ("material_wrench_world", 6, "N, N m", "pending material reaction wrench (oracle; not fed back in workload A)"),
    ("invalid_flag", 1, "0/1", "device validity flag (nonfinite state / slot overflow)"),
)
# Name tokens that mark material, cut, contact, force or generator truth. A policy term whose name contains
# one of them (as a token or as a substring for the PRIVILEGED_KEYS names) is rejected.
FORBIDDEN_TOKENS = frozenset({
    "material", "code", "codes", "contact", "contacts", "wrench", "force", "forces", "torque", "reaction",
    "impulse", "stress", "oracle", "privileged", "particle", "particles", "component", "components", "label",
    "labels", "bond", "bonds", "severed", "progress", "truth", "bump", "bumps", "protrusion", "protrusions",
    "geometry", "mass", "positions", "velocities", "grasp", "surface", "hermite", "cut_events", "slot",
    "invalid", "reward"})


class PrivilegedObservationLeak(ValueError):
    """A privileged quantity would reach the deployable policy observation."""


def _tokens(name):
    return set(t for t in re.split(r"[^a-z0-9]+", name.lower()) if t)


def privileged_name_reasons(name):
    """Why ``name`` is privileged (empty list if deployable)."""
    lower = name.lower()
    reasons = []
    if name in PRIVILEGED_KEYS:
        reasons.append("policy_env.PRIVILEGED_KEYS member")
    reasons += [f"contains PRIVILEGED_KEYS name {key!r}" for key in sorted(PRIVILEGED_KEYS) if key in lower]
    reasons += [f"token {tok!r}" for tok in sorted(_tokens(name) & FORBIDDEN_TOKENS)]
    return reasons


def assert_policy_terms_deployable(terms):
    """Raise :class:`PrivilegedObservationLeak` if any policy term name is privileged."""
    leaks = {name: privileged_name_reasons(name) for name, *_ in terms}
    leaks = {name: why for name, why in leaks.items() if why}
    if leaks:
        raise PrivilegedObservationLeak(f"privileged quantities in the deployable policy block: {leaks}")
    return True


def _layout(terms):
    out, start = [], 0
    for name, dim, units, description in terms:
        if not isinstance(dim, int) or dim < 1:
            raise ValueError(f"term {name!r} needs a positive int dim")
        out.append({"name": name, "dim": dim, "slice": [start, start+dim], "units": units,
                    "description": description})
        start += dim
    if len({t["name"] for t in out}) != len(out):
        raise ValueError("duplicate observation term names")
    return out, start


class StateObservation:
    """Fixed-layout ``[N, D]`` float32 observation blocks on one device."""

    def __init__(self, num_envs, *, max_steps, control_dt, profile="state", device="cpu",
                 critic_terms=DEFAULT_CRITIC_TERMS):
        if profile not in PROFILES:
            raise ValueError(f"observation_profile must be one of {PROFILES}")
        if not isinstance(max_steps, int) or max_steps < 1 or not control_dt > 0:
            raise ValueError("max_steps must be a positive int and control_dt positive")
        assert_policy_terms_deployable(POLICY_TERMS)
        self.num_envs, self.max_steps, self.control_dt = int(num_envs), int(max_steps), float(control_dt)
        self.profile, self.device = profile, device
        self.policy_layout, self.policy_dim = _layout(POLICY_TERMS)
        self.critic_terms = tuple(critic_terms) if profile == "privileged_state" else ()
        self.critic_layout, critic_extra = _layout(self.critic_terms) if self.critic_terms else ([], 0)
        overlap = {t["name"] for t in self.policy_layout} & {t["name"] for t in self.critic_layout}
        if overlap:
            raise ValueError(f"critic terms repeat policy terms: {sorted(overlap)}")
        self.critic_dim = self.policy_dim+critic_extra if profile == "privileged_state" else 0

    def metadata(self):
        meta = {"version": OBSERVATION_VERSION, "profile": self.profile, "dtype": "float32",
                "policy": {"dim": self.policy_dim, "terms": self.policy_layout, "deployable": True}}
        if self.profile == "privileged_state":
            shifted = [{**t, "slice": [t["slice"][0]+self.policy_dim, t["slice"][1]+self.policy_dim]}
                       for t in self.critic_layout]
            meta["critic"] = {"dim": self.critic_dim, "deployable": False,
                              "terms": [{"name": "policy", "dim": self.policy_dim, "slice": [0, self.policy_dim],
                                         "description": "the deployable policy block, unchanged"}, *shifted]}
        return meta

    def compute(self, *, commanded_pose, step_count, previous_action, armed, privileged=None):
        """Assemble ``{"policy": [N,Dp]}`` (+ ``"critic"`` under ``privileged_state``) as float32 device tensors.

        ``privileged`` maps every critic term name to an ``[N]`` or ``[N, dim]`` tensor; it must be None under
        the ``state`` profile.
        """
        import torch
        N = self.num_envs
        if privileged is not None and self.profile != "privileged_state":
            raise PrivilegedObservationLeak("privileged inputs given under the deployable 'state' profile")
        if commanded_pose.shape != (N, 7) or previous_action.shape != (N, 8) or step_count.shape != (N,) \
                or armed.shape != (N,):
            raise ValueError("expected commanded_pose [N,7], previous_action [N,8], step_count [N], armed [N]")
        pose = commanded_pose.to(torch.float64)
        quat = torch.where((pose[:, 6:7] < 0), -pose[:, 3:], pose[:, 3:])
        steps = step_count.to(torch.float64)
        policy = torch.cat([pose[:, :3], quat, ((self.max_steps-steps)*self.control_dt)[:, None],
                            previous_action.to(torch.float64), armed.to(torch.float64)[:, None],
                            (steps/self.max_steps)[:, None]], 1).to(torch.float32)
        obs = {"policy": policy}
        if self.profile == "privileged_state":
            if privileged is None or set(privileged) != {name for name, *_ in self.critic_terms}:
                raise ValueError(f"privileged_state needs exactly the critic terms "
                                 f"{[name for name, *_ in self.critic_terms]}")
            parts = [policy]
            for term in self.critic_layout:
                value = privileged[term["name"]].to(torch.float32)
                value = value.reshape(N, -1)
                if value.shape != (N, term["dim"]):
                    raise ValueError(f"critic term {term['name']!r} must be [N,{term['dim']}]")
                parts.append(value)
            obs["critic"] = torch.cat(parts, 1)
        assert_observation_structure(obs, self.profile)
        return obs


def assert_observation_structure(obs, profile):
    """Runtime guard: only ``policy`` (+ ``critic`` with ``privileged_state``); no PRIVILEGED_KEYS anywhere."""
    allowed = {"policy", "critic"} if profile == "privileged_state" else {"policy"}
    if set(obs) - allowed or "policy" not in obs:
        raise PrivilegedObservationLeak(f"observation keys {sorted(obs)} not allowed for profile {profile!r}")
    from ..policy_env import find_privileged_keys
    found = find_privileged_keys({k: v for k, v in obs.items() if k != "critic"})
    if found:
        raise PrivilegedObservationLeak(f"privileged keys inside the policy observation: {found}")
    return True
