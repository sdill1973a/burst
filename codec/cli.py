"""burst-codec command: `burst-codec selftest [--golden]` (coder + refusal tests; --golden needs the model) or
`burst-codec repro <args>` (the README's reproduction commands, needs the model)."""
import runpy, sys

import burst_codec  # noqa: F401  (puts the flat codec modules on the path)

CMDS = {"selftest": "test_det_codec", "repro": "repro"}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print("usage: burst-codec selftest [--golden] | burst-codec repro <args>  (see README)")
        sys.exit(2)
    mod = CMDS[sys.argv[1]]
    sys.argv = [mod + ".py", *sys.argv[2:]]
    runpy.run_module(mod, run_name="__main__")


if __name__ == "__main__":
    main()
