"""DM0.5 fixed-base BF16 execution (SM80).

The project owns model math and sampling. Only host transforms are supplied by
the explicitly pinned, read-only OpenDM checkout; no external inference engine
is called. See docs/dm05_usage.md for the intentionally narrow first profile.
"""
