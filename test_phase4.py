"""
test_phase4.py
────────────────────────────────────────────────────────────────────────────
Phase 4 end-to-end demo — Enrolment + Authentication.

Run from project root:

    # Step 1: Enrol yourself
    python test_phase4.py --mode enroll --user_id your_name

    # Step 2: Authenticate
    python test_phase4.py --mode auth --user_id your_name

    # See all enrolled users
    python test_phase4.py --mode list

PHASE 4 COMPLETION TEST
────────────────────────
Enrolment:
  - Window opens, 3 second countdown
  - Captures 30 frames automatically
  - Shows progress bar
  - Prints "Enrolled successfully"

Authentication:
  - Window opens showing CNN score
  - Press SPACE → flash challenge
  - If LIVE: face matching runs
  - Shows ACCEPT or REJECT with similarity score

Phase 4 is complete when:
  ✓ Enrolled face returns ACCEPT with similarity > 0.50
  ✓ Different person returns REJECT with similarity < 0.50
  ✓ Phone photo returns REJECT at liveness stage (before matching)
────────────────────────────────────────────────────────────────────────────
"""

import argparse
import sys

from infrastructure.logger import get_logger
from verification.identity_store import IdentityStore

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="VeraVisage Phase 4 — Enrolment + Authentication"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["enroll", "auth", "list", "delete"],
        help="Operation mode",
    )
    parser.add_argument(
        "--user_id",
        default=None,
        help="User identifier (required for enroll, auth, delete)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device (default: cuda)",
    )
    parser.add_argument(
        "--camera",
        default=0,
        type=int,
        help="Webcam index (default: 0)",
    )
    parser.add_argument(
        "--threshold",
        default=0.50,
        type=float,
        help="Similarity threshold for authentication (default: 0.50)",
    )
    args = parser.parse_args()

    # ── LIST ──────────────────────────────────────────────────────────────
    if args.mode == "list":
        store = IdentityStore()
        users = store.list_users()
        if not users:
            print("\n  No enrolled users found.")
            print("  Run: python test_phase4.py --mode enroll --user_id <name>\n")
        else:
            print(f"\n  Enrolled users ({len(users)}):")
            for u in users:
                print(f"    - {u}")
            print()
        return

    # ── DELETE ────────────────────────────────────────────────────────────
    if args.mode == "delete":
        if not args.user_id:
            print("ERROR: --user_id required for delete mode")
            sys.exit(1)
        store = IdentityStore()
        try:
            store.delete(args.user_id)
            print(f"\n  Deleted enrollment for '{args.user_id}'\n")
        except Exception as e:
            print(f"\n  Error: {e}\n")
        return

    # ── ENROLL ────────────────────────────────────────────────────────────
    if args.mode == "enroll":
        if not args.user_id:
            print("ERROR: --user_id required for enroll mode")
            sys.exit(1)

        from pipeline.enroll_pipeline import run_enroll

        print(f"\n  Starting enrolment for: {args.user_id}")
        print("  Look directly at your webcam.\n")

        result = run_enroll(
            user_id=args.user_id,
            device=args.device,
            camera_index=args.camera,
            show_preview=True,
        )

        print("\n" + "=" * 50)
        if result["status"] == "enrolled":
            print(f"  ✓ ENROLLED SUCCESSFULLY")
            print(f"  User:         {result['user_id']}")
            print(f"  Frames used:  {result['frames_used']}")
            print(f"  {result['reason']}")
            print()
            print(f"  Now run authentication:")
            print(
                f"  python test_phase4.py --mode auth "
                f"--user_id {result['user_id']}"
            )
        else:
            print(f"  ✗ ENROLMENT FAILED")
            print(f"  User:    {result['user_id']}")
            print(f"  Reason:  {result['reason']}")
        print("=" * 50 + "\n")
        return

    # ── AUTH ──────────────────────────────────────────────────────────────
    if args.mode == "auth":
        if not args.user_id:
            print("ERROR: --user_id required for auth mode")
            sys.exit(1)

        from pipeline.auth_pipeline import run_authentication

        print(f"\n  Starting authentication for: {args.user_id}")
        print("  Press SPACE to trigger the flash challenge.\n")

        result = run_authentication(
            user_id=args.user_id,
            device=args.device,
            camera_index=args.camera,
            similarity_threshold=args.threshold,
        )

        print("\n" + "=" * 50)
        if result["accepted"]:
            print(f"  ✓ ACCESS GRANTED")
        else:
            print(f"  ✗ ACCESS DENIED")

        print(f"  User:             {result['user_id']}")
        print(f"  Liveness:         {'PASS' if result['liveness_passed'] else 'FAIL'}")
        print(f"  Liveness score:   {result['liveness_score']:.3f}")
        print(f"  Similarity:       {result['similarity']:.3f}")
        print(f"  Reason:           {result['reason']}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Interrupted.\n")
        sys.exit(0)