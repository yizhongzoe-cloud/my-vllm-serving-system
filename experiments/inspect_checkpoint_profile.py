#!/usr/bin/env python3
"""
Inspect and validate checkpoint cost profile.

Checks profile JSON structure, validates monotonicity, and allows
interactive inspection of checkpoint decisions under various (L, u, S, ΔS) scenarios.
"""

import argparse
import json
from pathlib import Path


class ProfileInspector:
    """Load, validate, and inspect checkpoint cost profile."""

    def __init__(self, profile_path: str):
        with open(profile_path) as f:
            self.profile = json.load(f)

        self.prefill_data = {int(k): v for k, v in self.profile["prefill_ms_by_tokens"].items()}
        self.load_data = {int(k): v for k, v in self.profile["load_ms_by_bytes"].items()}
        self.ckpt_data = {int(k): v for k, v in self.profile["checkpoint_ms_by_bytes"].items()}
        self.c0 = self.profile["publication_overhead_ms"]

        # Sort for interpolation
        self.prefill_sorted = sorted(self.prefill_data.items())
        self.load_sorted = sorted(self.load_data.items())
        self.ckpt_sorted = sorted(self.ckpt_data.items())

    def validate(self) -> bool:
        """Validate profile structure and measurement reality."""
        print("Validating profile...")

        errors = []
        warnings = []

        # Check if this is a real measurement or placeholder
        is_real = self.profile.get("meta", {}).get("is_real_measurement", False)
        notes = self.profile.get("meta", {}).get("notes", "")
        if not is_real:
            errors.append(
                f"Profile contains PLACEHOLDER data (not real measurements): {notes}"
            )

        # Check structure
        required_keys = ["meta", "prefill_ms_by_tokens", "load_ms_by_bytes",
                         "checkpoint_ms_by_bytes", "publication_overhead_ms"]
        for key in required_keys:
            if key not in self.profile:
                errors.append(f"Missing key: {key}")

        # Check data types
        if self.c0 < 0:
            errors.append(f"publication_overhead_ms must be >= 0, got {self.c0}")

        # Check monotonicity
        prev = None
        for n, lat in self.prefill_sorted:
            if prev is not None and lat < prev:
                errors.append(f"prefill not monotonic: {prev} -> {lat} at n={n}")
            prev = lat

        prev = None
        for s, lat in self.load_sorted:
            if prev is not None and lat < prev:
                errors.append(f"load not monotonic: {prev} -> {lat} at S={s}")
            prev = lat

        prev = None
        for s, lat in self.ckpt_sorted:
            if prev is not None and lat < prev:
                errors.append(f"checkpoint not monotonic: {prev} -> {lat} at S={s}")
            prev = lat

        # Check prefill is convex (second derivative >= 0)
        if len(self.prefill_sorted) >= 3:
            # Compute slopes between consecutive points
            slopes = []
            for i in range(len(self.prefill_sorted) - 1):
                n1, t1 = self.prefill_sorted[i]
                n2, t2 = self.prefill_sorted[i + 1]
                if n2 != n1:
                    slope = (t2 - t1) / (n2 - n1)
                    slopes.append(slope)

            # Check second differences (should be >= 0 for convexity)
            has_negative_second_diff = False
            min_second_diff = 0
            for i in range(len(slopes) - 1):
                second_diff = slopes[i + 1] - slopes[i]
                if second_diff < -1e-6:  # Allow small numerical error
                    has_negative_second_diff = True
                    min_second_diff = min(min_second_diff, second_diff)

            if has_negative_second_diff:
                warnings.append(
                    f"prefill may not be convex (min 2nd derivative={min_second_diff:.6f})"
                )

        if errors:
            print("  ❌ Validation failed:")
            for err in errors:
                print(f"    - {err}")
            return False

        if warnings:
            print("  ⚠️  Validation passed with warnings:")
            for warn in warnings:
                print(f"    - {warn}")
            return True

        print("  ✅ Validation passed")
        return True

    def interpolate_linear(self, x: int, x_data: list[tuple[int, float]]) -> float:
        """Linear interpolation with clamping."""
        if not x_data:
            return 0.0

        if x <= x_data[0][0]:
            return x_data[0][1]
        if x >= x_data[-1][0]:
            return x_data[-1][1]

        # Find surrounding points
        for i in range(len(x_data) - 1):
            x1, y1 = x_data[i]
            x2, y2 = x_data[i + 1]
            if x1 <= x <= x2:
                # Linear interpolation
                t = (x - x1) / (x2 - x1) if x2 != x1 else 0
                return y1 + t * (y2 - y1)

        return x_data[-1][1]

    def t_prefill(self, n_tokens: int) -> float:
        """Get T_prefill(n) via linear interpolation."""
        return self.interpolate_linear(n_tokens, self.prefill_sorted)

    def t_load(self, n_bytes: int) -> float:
        """Get T_load(S) via linear interpolation."""
        return self.interpolate_linear(n_bytes, self.load_sorted)

    def t_ckpt(self, n_bytes: int) -> float:
        """Get T_ckpt(S) via linear interpolation."""
        return self.interpolate_linear(n_bytes, self.ckpt_sorted)

    def check_should_publish(
        self,
        L: int,  # published tokens
        u: int,  # unpublished stable tokens
        S: int,  # published bytes
        delta_S: int,  # unpublished bytes
        lambda_: float = 1.0,
    ) -> tuple[bool, dict]:
        """
        Check if checkpoint should be published under new profile-driven rule.

        Rule: T_replay(L, u) > T_load(ΔS) + λ * T_ckpt(ΔS) + c0

        Returns:
            (should_publish, details)
        """
        replay_cost = self.t_prefill(L + u) - self.t_prefill(L)
        load_cost = self.t_load(S + delta_S) - self.t_load(S)
        ckpt_cost = self.t_ckpt(delta_S)

        total_cost = load_cost + lambda_ * ckpt_cost + self.c0

        should_publish = replay_cost > total_cost

        return should_publish, {
            "L": L,
            "u": u,
            "S": S,
            "ΔS": delta_S,
            "T_replay(L,u)": replay_cost,
            "T_load(ΔS)": load_cost,
            "T_ckpt(ΔS)": ckpt_cost,
            "c0": self.c0,
            "total_cost": total_cost,
            "should_publish": should_publish,
            "margin": replay_cost - total_cost,
        }

    def scan_typical_workloads(self, lambda_: float = 1.0):
        """Scan typical workload scenarios."""
        print(f"\nScanning typical workloads (λ={lambda_})...")
        print("=" * 100)

        scenarios = [
            ("W1 Short (100 tokens, small checkpoint)", dict(L=64, u=[64, 128], S=131072, delta_S=[131072])),
            ("W2 Long (512 tokens, medium checkpoint)", dict(L=256, u=[256, 512], S=524288, delta_S=[262144, 524288])),
            ("W3 Bursty (256 tokens, variable checkpoint)", dict(L=128, u=[128, 256, 512], S=262144, delta_S=[131072, 262144])),
        ]

        for scenario_name, params in scenarios:
            print(f"\n{scenario_name}")
            L = params["L"]
            u_list = params["u"]
            S = params["S"]
            delta_S_list = params["delta_S"]

            for u in u_list:
                for delta_S in delta_S_list:
                    should_pub, details = self.check_should_publish(L, u, S, delta_S, lambda_)

                    status = "✅ PUBLISH" if should_pub else "⏸️  HOLD"
                    margin = details["margin"]
                    print(f"  u={u:4d}, ΔS={delta_S:7d}B: {status:15s} (margin={margin:+8.2f}ms)")

        print("\n" + "=" * 100)

    def print_summary(self):
        """Print profile summary."""
        print("\nProfile Summary")
        print("=" * 70)
        print(f"Model: {self.profile['meta']['model']}")
        print(f"Device: {self.profile['meta']['device']}")
        print(f"Block size: {self.profile['meta']['block_size_tokens']} tokens")
        print(f"Publication overhead (c0): {self.c0:.2f} ms")
        print()
        print(f"T_prefill: {len(self.prefill_sorted)} points")
        print(f"  Range: {self.prefill_sorted[0][0]}-{self.prefill_sorted[-1][0]} tokens")
        print(f"  Latency: {self.prefill_sorted[0][1]:.2f}-{self.prefill_sorted[-1][1]:.2f} ms")
        print()
        print(f"T_load: {len(self.load_sorted)} points")
        print(f"  Range: {self.load_sorted[0][0]}-{self.load_sorted[-1][0]} bytes")
        print(f"  Latency: {self.load_sorted[0][1]:.2f}-{self.load_sorted[-1][1]:.2f} ms")
        print()
        print(f"T_ckpt: {len(self.ckpt_sorted)} points (size-dependent only)")
        print(f"  Range: {self.ckpt_sorted[0][0]}-{self.ckpt_sorted[-1][0]} bytes")
        print(f"  Latency: {self.ckpt_sorted[0][1]:.2f}-{self.ckpt_sorted[-1][1]:.2f} ms")
        print()


def main():
    parser = argparse.ArgumentParser(description="Inspect checkpoint cost profile")
    parser.add_argument("profile_path", help="Path to checkpoint_cost_profile.json")
    parser.add_argument("--lambda", type=float, default=1.0, dest="lambda_",
                        help="Weight λ on checkpoint cost (default: 1.0)")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation")
    parser.add_argument("--skip-workloads", action="store_true", help="Skip typical workload scan")
    args = parser.parse_args()

    print("=" * 70)
    print("Checkpoint Profile Inspector")
    print("=" * 70)

    # Load profile
    inspector = ProfileInspector(args.profile_path)

    # Print summary
    inspector.print_summary()

    # Validate
    if not args.skip_validation:
        if not inspector.validate():
            return 1

    # Scan typical workloads
    if not args.skip_workloads:
        inspector.scan_typical_workloads(lambda_=args.lambda_)

    # Interactive mode
    print("\nInteractive mode: enter (L u S ΔS) to check decision, or 'quit' to exit")
    print("Example: 128 256 262144 131072")
    while True:
        try:
            line = input("\n> ").strip()
            if line.lower() in ("quit", "exit", "q"):
                break

            parts = line.split()
            if len(parts) != 4:
                print("Expected 4 integers: L u S ΔS")
                continue

            L, u, S, delta_S = map(int, parts)
            should_pub, details = inspector.check_should_publish(L, u, S, delta_S, args.lambda_)

            print()
            for key, value in details.items():
                if isinstance(value, float):
                    print(f"  {key:20s} = {value:+10.2f}")
                else:
                    print(f"  {key:20s} = {value}")

            status = "✅ PUBLISH" if should_pub else "⏸️  HOLD"
            print(f"\n  Decision: {status}")

        except KeyboardInterrupt:
            break
        except ValueError as e:
            print(f"Invalid input: {e}")

    print("\nDone.")


if __name__ == "__main__":
    exit(main() or 0)
