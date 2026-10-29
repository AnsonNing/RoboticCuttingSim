"""Read-only BladeMaster demo evaluators (ADR 0007 section 4).

Every function only reads its inputs; recordings are never modified. These
reports are separate from the frozen v1 oracle (``tasks.evaluate_task_oracle``)
and from Evaluation v2: they add the swept-plane interface statistic, the
label-to-swept-plane distance distribution, TRIM separation (reported apart
from the surface precision score) and the post-cut manipulation check of
SLICE-PUSH-01. Oracle material state here is evaluation-only, never an
observation.
"""
import numpy as np

from .cut_topology import mechanical_components
from .multi_cut_evaluation import evaluate_multi_cut

PROVENANCE = "Recomputed actual eight-node compatibility graph including positive weights, stress gradients and code-zero bridges"


def _plane(origin, normal):
    origin, normal = np.asarray(origin, float), np.asarray(normal, float)
    norm = np.linalg.norm(normal)
    if origin.shape != (3,) or normal.shape != (3,) or not np.isfinite([*origin, *normal]).all() or norm <= 0:
        raise ValueError("A plane needs a finite origin and a nonzero finite normal")
    return origin, normal/norm


def swept_plane_interface(nodes, weights, gradients, labels, reference_positions, plane_origin, plane_normal):
    """The frozen ``_interface_location`` midpoint statistic against any plane.

    Same logic as ``tasks._interface_location`` (actual MPM support including
    zero-weight stress-gradient entries; per cross-component node, the
    midpoints of the per-side signed-distance extrema), but the signed
    distance is measured to an arbitrary plane, e.g. the plane the engaged
    edge actually swept. ``reference_positions`` and the plane share a frame.
    """
    origin, normal = _plane(plane_origin, plane_normal)
    nodes, weights, gradients = np.asarray(nodes), np.asarray(weights), np.asarray(gradients)
    labels, reference = np.asarray(labels), np.asarray(reference_positions, float)
    support = ((weights > 0) | (np.linalg.norm(gradients, axis=-1) > 0)).ravel()
    particle = np.repeat(np.arange(len(labels)), nodes.shape[1])[support]
    node = nodes.ravel()[support]
    order = np.argsort(node, kind="stable")
    groups = np.split(particle[order], np.flatnonzero(np.diff(node[order]))+1)
    signed = (reference-origin)@normal
    maximum, incidences, interfaces = 0., 0, 0
    for group in groups:
        occupied = np.unique(labels[group])
        if len(occupied) < 2:
            continue
        interfaces += 1
        for i, first in enumerate(occupied):
            a = signed[group[labels[group] == first]]
            for second in occupied[i+1:]:
                b = signed[group[labels[group] == second]]
                maximum = max(maximum, abs((a.min()+b.min())/2), abs((a.max()+b.max())/2))
                incidences += len(a)*len(b)
    return {"cross_component_reference_node_count": interfaces,
            "severed_particle_pair_node_incidences": incidences,
            "maximum_reference_midpoint_plane_distance_m": None if not interfaces else float(maximum),
            "plane_origin": origin.tolist(), "plane_normal": normal.tolist(),
            "statistic": "frozen _interface_location midpoint statistic measured to the given plane"}


def fit_swept_plane(edge_points, edge_lines, *, edge_half_length_m=None, reference_normal=None):
    """Least-squares plane traced by recorded edge poses.

    ``edge_points`` (T,3) are edge centres while the edge was engaged and
    ``edge_lines`` (T,3) the edge-line directions. Each pose contributes its
    centre and the two points ``centre +- L * line`` (L defaults to the RMS
    spread of the centres), so both the advance and the edge line lie in the
    fitted plane. Residuals are reported for the centres and the line tilt.
    """
    points, lines = np.asarray(edge_points, float), np.asarray(edge_lines, float)
    if (points.ndim != 2 or points.shape[1] != 3 or lines.shape != points.shape or len(points) < 2
            or not np.isfinite(points).all() or not np.isfinite(lines).all()):
        raise ValueError("Need at least two finite edge centres and matching edge-line directions")
    norms = np.linalg.norm(lines, axis=1)
    if np.any(norms <= 0):
        raise ValueError("Edge-line directions must be nonzero")
    lines = lines/norms[:, None]
    spread = float(np.sqrt(((points-points.mean(0))**2).sum(1).mean()))
    length = spread if edge_half_length_m is None else float(edge_half_length_m)
    if not length > 0:
        raise ValueError("The edge poses do not span a swept surface")
    samples = np.concatenate([points, points+length*lines, points-length*lines])
    origin = samples.mean(0)
    _, singular, vt = np.linalg.svd(samples-origin, full_matrices=False)
    if singular[1] <= 1e-9*max(singular[0], 1e-30):
        raise ValueError("Edge poses are degenerate (a line, not a swept plane)")
    normal = vt[2]
    if reference_normal is not None and normal@np.asarray(reference_normal, float) < 0:
        normal = -normal
    residual = (points-origin)@normal
    return {"origin": origin.tolist(), "normal": normal.tolist(), "samples": int(len(points)),
            "rms_residual_m": float(np.sqrt(np.mean(residual**2))), "max_residual_m": float(np.abs(residual).max()),
            "max_edge_line_out_of_plane_rad": float(np.arcsin(np.clip(np.abs(lines@normal), 0, 1)).max()),
            "edge_half_length_m": length, "scope": "Plane fitted to recorded edge poses; not a material measurement"}


def label_distance_distribution(reference_positions, codes, plane_origin, plane_normal, *, digit=None,
                                percentiles=(50, 90, 95, 99, 100)):
    """Distances of labelled particles (reference positions) to a swept plane.

    ``codes`` may be one frame (N,) or a history (T,N) (the last frame is used).
    With ``digit`` only that ternary digit counts as a label. Side agreement is
    the fraction whose digit value (1 or 2) matches the sign of the signed
    distance (1 <-> negative), reported both ways because the blade-normal sign
    convention is the caller's.
    """
    origin, normal = _plane(plane_origin, plane_normal)
    codes, reference = np.asarray(codes), np.asarray(reference_positions, float)
    if codes.ndim == 2:
        codes = codes[-1]
    if codes.shape != (len(reference),) or codes.dtype.kind not in "iu" or reference.shape != (len(codes), 3):
        raise ValueError("Need aligned integer codes and reference positions")
    value = codes if digit is None else (codes//3**int(digit)) % 3
    labelled = value != 0
    signed = (reference[labelled]-origin)@normal
    distance = np.abs(signed)
    result = {"labelled_particles": int(labelled.sum()), "digit": digit,
              "plane_origin": origin.tolist(), "plane_normal": normal.tolist()}
    if not labelled.any():
        return {**result, "percentiles_m": None, "maximum_m": None, "side_agreement_fraction": None}
    if digit is not None:
        agree = float(np.mean((value[labelled] == 1) == (signed < 0)))
        result["side_agreement_fraction"] = max(agree, 1.-agree)
        result["side_convention_digit1_negative"] = agree >= .5
    return {**result, "percentiles_m": {str(p): float(np.percentile(distance, p)) for p in percentiles},
            "maximum_m": float(distance.max()), "mean_m": float(distance.mean())}


def trim_separation(reference_positions, final_positions, masses, nodes, weights, gradients, final_codes, target, *,
                    cuts=1, heightfield=None, minimum_removed_fraction=.9, minimum_substrate_share=.99,
                    maximum_detached_below_fraction=.01):
    """TRIM separation report, separate from the surface precision score.

    Above-plane removal is exactly the frozen check
    (``material_task_metrics``: particle centres initially above the target,
    detached from the components holding the lowest reference particles).
    The substrate share is the largest component's share of the below-plane
    reference mass; detached below-plane mass is below-plane reference mass
    outside the frozen substrate components. Precision metrics are not part
    of this report. ``target`` is a ``scene_frames.WorkpieceTarget``.
    """
    from .scene_validation import material_task_metrics
    masses = np.asarray(masses, float)
    final_codes = np.asarray(final_codes)
    if final_codes.ndim == 2:
        final_codes = final_codes[-1]
    count, labels = mechanical_components(np.asarray(nodes), np.asarray(weights), final_codes, cuts,
                                          gradients=np.asarray(gradients))
    frozen = material_task_metrics(reference_positions, final_positions, masses, labels, PROVENANCE, target, heightfield)
    initial = target.points_to_local(reference_positions)
    below = initial[:, 2] <= target.height_m+frozen["reference_plane_tolerance_m"]
    below_mass = float(masses[below].sum())
    per_component = np.bincount(labels[below], weights=masses[below], minlength=count)
    substrate = frozen["substrate_mask_debug"]
    share = None if below_mass <= 0 else float(per_component.max()/below_mass)
    detached_below = None if below_mass <= 0 else float(masses[below & ~substrate].sum()/below_mass)
    removed = frozen["detached_initial_above_target_mass_fraction"]
    checks = {"reference_above_plane_material_removed": removed is not None and removed >= minimum_removed_fraction,
              "largest_component_holds_below_plane_mass": share is not None and share >= minimum_substrate_share,
              "detached_below_plane_mass_bounded": detached_below is not None and detached_below <= maximum_detached_below_fraction,
              "single_substrate_component": len(frozen["reference_substrate_component_ids"]) == 1}
    return {"protocol": "trim_separation_v1", "status": "PASS_TRIM_SEPARATION" if all(checks.values()) else "FAIL_TRIM_SEPARATION",
            "checks": {k: bool(v) for k, v in checks.items()},
            "metrics": {"mechanical_components": int(count), "removed_reference_above_plane_mass_fraction": removed,
                        "largest_component_below_plane_mass_share": share,
                        "detached_below_plane_mass_fraction": detached_below,
                        "reference_below_plane_mass_kg": below_mass,
                        "reference_substrate_component_ids": frozen["reference_substrate_component_ids"]},
            "thresholds": {"minimum_removed_fraction": minimum_removed_fraction, "minimum_substrate_share": minimum_substrate_share,
                           "maximum_detached_below_fraction": maximum_detached_below_fraction},
            "scope": "Separation only; the 0.5 mm surface precision thresholds are a separate score (frozen oracle)"}


def trim_separation_for_task(task, recording, **thresholds):
    """``trim_separation`` on a recorded WORLD episode of a flatten task."""
    from .scene_frames import WorkpieceTarget
    if task.goal_kind != "flatten":
        raise ValueError("TRIM separation needs a flatten task")
    target = WorkpieceTarget(task.world_from_workpiece_xyzw, task.target_origin_local_m[2], task.edge_clearance_m)
    positions = np.asarray(recording["positions"])
    return trim_separation(recording["reference_positions"], positions[-1], recording["mass_kg"],
                           recording["reference_stencil_nodes"], recording["reference_stencil_weights"],
                           recording["reference_stencil_gradients"], recording["codes"], target,
                           heightfield=task.geometry, **thresholds)


def _merged(later, earlier):
    """True when one later component holds particles of two earlier components."""
    pairs = np.unique(np.stack([later, earlier], 1), axis=0)
    return bool(len(np.unique(pairs[:, 0])) < len(pairs))


def post_cut_manipulation(time_s, positions, codes, masses, nodes, weights, gradients, push_start_s, push_end_s,
                          push_direction, pushed_selector, cuts=2, *, minimum_components=None,
                          minimum_pushed_displacement_m=.01, maximum_other_displacement_m=.002,
                          minimum_piece_mass_fraction=.05, minimum_selector_share=.5):
    """Post-cut push check on the actual stencil graph (read-only).

    Components come from ``mechanical_components`` with weights and stress
    gradients at the last frame at or before ``push_start_s``. Checks: at least
    ``cuts + 1`` (or ``minimum_components``) components; no code changes from
    that frame to the first frame at or after ``push_end_s``; the pushed
    component (largest mass overlap with the boolean ``pushed_selector``)
    moves its mass centroid at least ``minimum_pushed_displacement_m`` along
    ``push_direction``; every other piece (>= ``minimum_piece_mass_fraction``)
    moves at most ``maximum_other_displacement_m``; no later frame merges
    push-start components.
    """
    time, positions, codes = np.asarray(time_s, float), np.asarray(positions, float), np.asarray(codes)
    masses, nodes, weights, gradients = (np.asarray(masses, float), np.asarray(nodes), np.asarray(weights),
                                         np.asarray(gradients))
    selector = np.asarray(pushed_selector, bool)
    n = len(masses)
    if (time.ndim != 1 or len(time) < 2 or np.any(np.diff(time) <= 0) or positions.shape != (len(time), n, 3)
            or codes.shape != (len(time), n) or codes.dtype.kind not in "iu" or selector.shape != (n,)
            or nodes.shape[0] != n or weights.shape != nodes.shape or gradients.shape != (*nodes.shape, 3)
            or np.any(masses <= 0) or not np.isfinite(positions).all() or not np.isfinite(masses).all()):
        raise ValueError("Recording shapes, times, masses or selector are invalid")
    direction = np.asarray(push_direction, float)
    direction = direction/np.linalg.norm(direction)
    minimum_components = cuts+1 if minimum_components is None else minimum_components
    base = {"protocol": "post_cut_manipulation_v1", "push_interval_s": [float(push_start_s), float(push_end_s)],
            "scope": "Actual positive-weight/stress-gradient transfer graph; components are mechanical, not visual"}
    before = np.flatnonzero(time <= push_start_s+1e-9)
    after = np.flatnonzero(time >= push_end_s-1e-9)
    if not len(before) or not len(after) or after[0] <= before[-1]:
        return {**base, "status": "NOT_RUN", "checks": {}, "metrics": {},
                "missing_evidence": ["Recorded frames at or before push start and at or after push end"]}
    start, end = int(before[-1]), int(after[0])
    cache = {}

    def partition(index):
        key = codes[index].tobytes()
        if key not in cache:
            cache[key] = mechanical_components(nodes, weights, codes[index], cuts, gradients=gradients)
        return cache[key]

    count, labels = partition(start)
    end_count, _ = partition(end)
    total = masses.sum()
    component_mass = np.bincount(labels, weights=masses, minlength=count)

    def centroid(frame):
        return np.stack([np.bincount(labels, weights=masses*positions[frame, :, axis], minlength=count)
                         for axis in range(3)], 1)/component_mass[:, None]

    displacement = centroid(end)-centroid(start)
    final_displacement = centroid(len(time)-1)-centroid(start)
    overlap = np.bincount(labels[selector], weights=masses[selector], minlength=count)
    pushed = int(np.argmax(overlap))
    selector_share = float(overlap[pushed]/masses[selector].sum()) if selector.any() else 0.
    pieces = np.flatnonzero(component_mass/total >= minimum_piece_mass_fraction)
    others = [int(c) for c in pieces if c != pushed]
    other_motion = [float(np.linalg.norm(displacement[c])) for c in others]
    changed = int(np.count_nonzero(np.any(codes[start:end+1] != codes[start], axis=1)))
    merges = [float(time[f]) for f in range(start+1, len(time)) if _merged(partition(f)[1], labels)]
    along = float(displacement[pushed]@direction)
    checks = {"components_after_cuts": count >= minimum_components,
              "no_label_changes_during_push": changed == 0,
              "pushed_piece_identified": selector_share >= minimum_selector_share
                  and component_mass[pushed]/total >= minimum_piece_mass_fraction,
              "pushed_piece_displaced": along >= minimum_pushed_displacement_m,
              "other_pieces_stationary": all(d <= maximum_other_displacement_m for d in other_motion),
              "components_never_merge": not merges}
    metrics = {"push_start_frame_time_s": float(time[start]), "push_end_frame_time_s": float(time[end]),
               "components_at_push_start": int(count), "components_at_push_end": int(end_count),
               "component_mass_fractions": (component_mass/total).tolist(), "pushed_component": pushed,
               "pushed_selector_mass_share": selector_share,
               "pushed_displacement_along_push_m": along,
               "pushed_displacement_m": displacement[pushed].tolist(),
               "other_piece_components": others, "other_piece_displacement_norm_m": other_motion,
               "final_frame_displacement_norm_m": np.linalg.norm(final_displacement, axis=1).tolist(),
               "frames_with_code_changes_during_push": changed, "merge_frame_times_s": merges}
    status = "PASS_POST_CUT_MANIPULATION" if all(checks.values()) else "FAIL_POST_CUT_MANIPULATION"
    return {**base, "status": status, "checks": {k: bool(v) for k, v in checks.items()}, "metrics": metrics,
            "thresholds": {"minimum_components": minimum_components,
                           "minimum_pushed_displacement_m": minimum_pushed_displacement_m,
                           "maximum_other_displacement_m": maximum_other_displacement_m,
                           "minimum_piece_mass_fraction": minimum_piece_mass_fraction,
                           "minimum_selector_share": minimum_selector_share}}


def slice_push_report(task, recording):
    """SLICE-PUSH-01 outcome: evaluate_multi_cut (declared planes) + post_cut_manipulation.

    ``recording`` is a WORLD episode (time_s, positions, codes, mass_kg,
    reference_positions, reference_indices, reference stencils). The push
    interval comes from ``recording["push_interval_s"]`` when the runner
    recorded the actual push phase, else from the task's reference schedule
    (the source is reported). The frozen two-part oracle is NOT_APPLICABLE.
    """
    required = ("time_s", "positions", "codes", "mass_kg", "reference_positions", "reference_indices",
                "reference_stencil_nodes", "reference_stencil_weights", "reference_stencil_gradients")
    base = {"task_id": task.task_id, "protocol": "slice_push_report_v1",
            "protocol_id": task.protocol_document()["protocol_id"], "frozen_two_part_oracle": "NOT_APPLICABLE",
            "full_simulation_acceptance": False, "policy_observation": False}
    missing = [key for key in required if key not in recording]
    if missing:
        return {**base, "status": "NOT_RUN", "missing_evidence": missing}
    limits = task.thresholds
    goal = task.multi_cut_goal()
    multi = evaluate_multi_cut(recording, goal)
    if "push_interval_s" in recording:
        push, source = tuple(float(v) for v in recording["push_interval_s"]), "recorded_push_phase"
    else:
        push, source = tuple(task.push_interval_s), "task_reference_schedule"
    reference = np.asarray(recording["reference_positions"], float)
    manipulation_limits = limits["post_cut_manipulation"]
    manipulation = post_cut_manipulation(
        recording["time_s"], recording["positions"], recording["codes"], recording["mass_kg"],
        recording["reference_stencil_nodes"], recording["reference_stencil_weights"],
        recording["reference_stencil_gradients"], push[0], push[1], task.push_direction_world(),
        task.pushed_piece_mask(reference), cuts=task.max_cuts,
        minimum_components=manipulation_limits["minimum_components_after_cuts"],
        minimum_pushed_displacement_m=manipulation_limits["minimum_pushed_piece_displacement_along_push_m"],
        maximum_other_displacement_m=manipulation_limits["maximum_other_piece_displacement_m"],
        minimum_piece_mass_fraction=manipulation_limits["minimum_piece_mass_fraction"],
        minimum_selector_share=manipulation_limits["minimum_pushed_selector_mass_share"])
    distances = [label_distance_distribution(reference, recording["codes"], origin, normal, digit=index)
                 for index, (origin, normal) in enumerate(task.cut_planes_world())]
    statuses = (multi["status"], manipulation["status"])
    status = ("PASS_SLICE_PUSH" if statuses == ("PASS_MULTI_CUT_OUTCOME", "PASS_POST_CUT_MANIPULATION")
              else "FAIL_SLICE_PUSH" if any(s.startswith("FAIL") for s in statuses) else "NOT_RUN")
    return {**base, "status": status, "multi_cut": multi, "post_cut_manipulation": manipulation,
            "push_interval_source": source, "label_distance_to_declared_planes": distances,
            "multi_cut_goal": {"planes_world": [list(map(list, plane)) for plane in goal.planes],
                               "minimum_part_mass_fraction": goal.minimum_part_mass_fraction,
                               "minimum_main_parts_mass_fraction": goal.minimum_main_parts_mass_fraction,
                               "plane_error_bound_m": goal.plane_error_bound_m}}
