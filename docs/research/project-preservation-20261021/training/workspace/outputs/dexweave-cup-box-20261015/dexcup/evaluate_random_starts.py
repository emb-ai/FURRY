"""Random-scene evaluation must use the native neutral robot reset.

The former reference-frame initialization is retained only in
``evaluate_reference_pose_diagnostic`` for historical diagnostics.
"""
from .evaluate_full_reset import main

if __name__ == '__main__':
    main()
