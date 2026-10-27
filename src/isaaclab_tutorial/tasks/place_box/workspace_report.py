"""Population-specific acceptance and coverage; never combine easy and hard starts."""

import math


def wilson(successes, count):
    if not count:
        return [0.0, 1.0]
    z = 1.959963984540054
    rate = successes / count
    scale = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / scale
    half = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / scale
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize(episodes, population, position_noise, yaw_noise, policy, expected_per_case=15):
    """Return acceptance for one population, with explicit finite-sample coverage."""
    if population not in ("grid", "random", "stage", "nominal", "regression"):
        raise ValueError("Unknown audit population")
    if policy not in ("teacher", "camera"):
        raise ValueError("policy must be teacher or camera")
    source = {"grid": "grid", "regression": "replay"}.get(population, "current")
    if any(row.get("reset_source") != source for row in episodes):
        raise ValueError("Unexpected reset source: training-mixture contamination or missing provenance")
    n = len(episodes)
    successes = sum(row["success"] for row in episodes)
    rate = successes / n if n else 0.0
    threshold = 0.95 if population == "stage" else 0.99
    if population == "nominal":
        threshold = 0.999 if policy == "teacher" else 0.995
    if population == "regression":
        threshold = 1.0
    result = dict(
        population=population,
        policy=policy,
        episodes=n,
        successes=successes,
        success_rate=rate,
        success_wilson_95=wilson(successes, n),
        required_success_rate=threshold,
    )
    sufficient = n >= {"nominal": 3072, "regression": 3, "stage": 128}.get(population, 1)
    groups = sorted({r["seed_group"] for r in episodes if "seed_group" in r})
    result["seed_groups"] = groups
    if population in ("grid", "random", "nominal"):
        sufficient &= len(groups) >= 3
    result["stage_counts"] = {
        key: sum(bool(row.get(key, False)) for row in episodes)
        for key in ("grasped", "lifted", "placed", "success", "timed_out", "dropped", "out_of_bounds", "unstable_robot")
    }
    passed = rate >= threshold
    if population in ("grid", "random", "stage"):
        if position_noise <= 0 or yaw_noise <= 0:
            raise ValueError("Spatial/yaw audit ranges must be positive")
        counts = [[0] * 5 for _ in range(5)]
        wins = [[0] * 5 for _ in range(5)]
        yaw_counts, yaw_wins = [0] * 8, [0] * 8
        case_counts, case_wins = [0] * 200, [0] * 200
        for row in episodes:
            x, y, _, qx, qy, qz, qw = row["initial_box_pose"]
            dx, dy, yaw = x - 0.17, y + 0.1, 2 * math.atan2(qz, qw)
            if not all(math.isfinite(v) for v in (x, y, qx, qy, qz, qw)):
                raise ValueError("Audit poses must be finite")
            if abs(dx) > position_noise + 1e-4 or abs(dy) > position_noise + 1e-4 or abs(yaw) > yaw_noise + 1e-5:
                raise ValueError("Saved initial pose is outside the requested reset distribution")
            ix = min(4, max(0, int((dx / position_noise + 1) * 2.5)))
            iy = min(4, max(0, int((dy / position_noise + 1) * 2.5)))
            iz = min(7, max(0, int((yaw / yaw_noise + 1) * 4 + 1e-6)))
            counts[iy][ix] += 1
            wins[iy][ix] += row["success"]
            yaw_counts[iz] += 1
            yaw_wins[iz] += row["success"]
            if population == "grid":
                if not isinstance(row.get("case_id"), int) or row["case_id"] < 0:
                    raise ValueError("Missing or invalid grid case index")
                case = row["case_id"] % 200
                expected = (
                    (case // 40 / 2 - 1) * position_noise,
                    ((case // 8) % 5 / 2 - 1) * position_noise,
                    (case % 8 / 4 - 1) * yaw_noise,
                )
                angle_error = abs((yaw - expected[2] + math.pi) % (2 * math.pi) - math.pi)
                if abs(dx - expected[0]) > 1e-4 or abs(dy - expected[1]) > 1e-4 or angle_error > 1e-5:
                    raise ValueError("Saved physical pose does not match its grid case index")
                case_counts[case] += 1
                case_wins[case] += row["success"]
        result.update(spatial_counts=counts, spatial_successes=wins, yaw_counts=yaw_counts, yaw_successes=yaw_wins)
        if population != "stage":
            sufficient &= min(min(row) for row in counts) >= 100 and min(yaw_counts) >= 100
            passed &= all(
                win / count >= 0.95
                for a, b in zip(wins, counts, strict=True)
                for win, count in zip(a, b, strict=True)
                if count
            )
            passed &= all(win / count >= 0.95 for win, count in zip(yaw_wins, yaw_counts, strict=True) if count)
        if population == "grid":
            sufficient &= all(count == expected_per_case for count in case_counts)
            identities = {(row.get("seed_group"), row["case_id"]) for row in episodes}
            sufficient &= len(identities) == len(episodes)
            expected_count = 200 * expected_per_case // len(groups) if groups else 0
            for group in groups:
                actual = {r["case_id"] for r in episodes if r.get("seed_group") == group}
                sufficient &= actual == set(range(expected_count))
            result.update(case_counts=case_counts, case_successes=case_wins)
    result["coverage_sufficient"] = bool(sufficient)
    result["status"] = "insufficient_coverage" if not sufficient else ("pass" if passed else "fail")
    return result
