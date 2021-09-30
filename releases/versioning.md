---
parent: Releases
nav_order: 1
---

# Versioning

Thrust has its own versioning system for releases, independent of the
  versioning scheme of the NVIDIA HPC SDK or the CUDA Toolkit.

Today, Thrust version numbers have a specific [semantic meaning](https://semver.org/).
Releases prior to 1.10.0 largely, but not strictly, followed these semantic
  meanings.

The version number for a Thrust release uses the following format:
  `MMM.mmm.ss-ppp`, where:

* `THRUST_VERSION_MAJOR`/`MMM`: Major version, up to 3 decimal digits.
  It is incremented when changes that are API-backwards-incompatible are made.
* `THRUST_VERSION_MINOR`/`mmm`: Minor version, up to 3 decimal digits.
  It is incremented when breaking API, ABI, or semantic changes are made.
* `THRUST_VERSION_SUBMINOR`/`ss`: Subminor version, up to 2 decimal digits.
  It is incremented when notable new features or bug fixes or features that are
  API-backwards-compatible are made.
* `THRUST_PATCH_NUMBER`/`ppp`: Patch number, up to 3 decimal digits.
  This is no longer used and will be zero for all future releases.

The `<thrust/version.h>` header defines `THRUST_*` macros for all of the
  version components mentioned above.
Additionally, a `THRUST_VERSION` macro is defined, which is an integer literal
  containing all of the version components except for `THRUST_PATCH_NUMBER`.

## Trunk Based Development

Thrust uses [trunk based development](https://trunkbaseddevelopment.com).
There is a single long-lived branch called `main`, which is public and the
  "source of truth".
All other branches are downstream from `main`.
Engineers may create branches for feature development.
Such branches always merge into `main`.
There are no release branches.
Releases are produced by taking a snapshot of `main` ("snapping").
After a release has been snapped from `main`, it will never be changed.

## Branches and Tags

The following tag names are used in the Thrust project:

* `nvhpc-X.Y`: the tag that directly corresponds to what has been
  shipped in the NVIDIA HPC SDK release X.Y.
* `cuda-X.Y`: the tag that directly corresponds to what has been shipped
  in the CUDA Toolkit release X.Y.
* `A.B.C`: the tag that directly corresponds to Thrust version A.B.C.
* `A.B.C-rcN`: the tag that directly corresponds to Thrust version A.B.C
  release candidate N.

The following branch names are used in the Thrust project:

* `main`: the "source of truth" development branch of Thrust.
* `old-master`: the old "source of truth" branch, before unification of
  public and internal repositories.
* `feature/<name>`: feature branch for a feature under development.
* `bug/<bug-system>/<bug-description>-<bug-id>`: bug fix branch, where
  `bug-system` is `github` or `nvidia`.

On the rare occasion that we cannot do work in the open, for example when
  developing a change specific to an unreleased product, these branches may
  exist on an internal NVIDIA GitLab instance instead of the public GitHub.
By default, everything should be in the open on GitHub unless there is a strong
  motivation for it to not be open.

