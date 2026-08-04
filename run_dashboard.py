"""Entrypoint for the TDA basket & clustering dashboard.

Run this file directly, e.g.:
    python run_dashboard.py
or via your editor's "Run" button -- unlike src/dashboard/app.py, this does
not rely on `-m` package invocation (src/dashboard/app.py uses relative
imports internally, which only work when imported as part of the `src`
package, not when executed directly as a script).
"""
from Src.dashboard.app import main

if __name__ == "__main__":
    main()
