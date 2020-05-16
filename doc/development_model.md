# Thrust Branching and Development Model

The following is a description of how the Thrust development teams approaches branching and release tagging. This
is a living document that will evolve as our process evolves.

Thrust is distributed in three ways:

   * On GitHub.
   * In the NVIDIA HPC SDK.
   * In the CUDA Toolkit.

## Trunk Based Development

Thrust uses [trunk based development](https://trunkbaseddevelopment.com). There is a single long-lived
branch called `master`. Engineers may create branches for feature development. such branches always
merge into `master`. There are no release branches. Releases are produced by taking a snapshot of
`master` ("snapping"). After a release has been snapped from `master`, it will never be changed.

## Repositories

As Thrust is developed both on GitHub and internally at NVIDIA, there's three main places where code lives:

   * The Source of Truth, the [public Thrust repository](https://github.com/thrust/thrust), referred to as
     `github` later in this document.
   * An internal GitLab repository, referred to as `gitlab` later in this document.
   * An internal Perforce repository, referred to as `perforce` later in this document.

## Versioning

Thrust has its own versioning system for releases, independent of the versioning scheme of the NVIDIA
HPC SDK or the CUDA Toolkit.

Today, Thrust version numbers have a specific [semantic meaning](https://semver.org/).
Releases prior to 1.10.0 largely, but not strictly, followed these semantic meanings.

The version number for a Thrust release uses the following format:
`MMM.mmm.ss-ppp`, where:

   * `THRUST_VERSION_MAJOR`/`MMM`: Major version, up to 3 decimal digits. It is incremented
     when the fundamental nature of the library evolves, leading to widespread changes across the
     entire library interface with no guarantee of API, ABI, or semantic compatibility with former
     versions.
   * `THRUST_VERISON_MINOR`/`mmm`: Minor version, up to 3 decimal digits. It is incremented when
     breaking API, ABI, or semantic changes are made.
   * `THRUST_VERSION_SUBMINOR`/`ss`: Subminor version, up to 2 decimal digits. It is incremented
     when notable new features or bug fixes or features that are API, ABI, and semantic backwards
     compatible are added.
   * `THRUST_PATCH_NUMBER`/`ppp`: Patch number, up to 3 decimal digits. It is incremented if any
     change in the repo whatsoever is made and no other version component has been incremented.

The `<thrust/version.h>` header defines `THRUST_*` macros for all of the version components mentioned
above. Additionally, a `THRUST_VERSION` macro is defined, which is an integer literal containing all
of the version components except for `THRUST_PATCH_NUMBER`

## Thrust Releases

| Thrust Release    | Included In                    |
| ----------------- | ------------------------------ |
| 1.9.10            | NVIDIA HPC SDK 20.5            |
| 1.9.9             | CUDA Toolkit 11.0              |
| 1.9.8-1           | NVIDIA HPC SDK 20.3            |
| 1.9.8             | CUDA Toolkit 11.0 Early Access |
| 1.9.7-1           | CUDA Toolkit 10.2 for Tegra    |
| 1.9.7             | CUDA Toolkit 10.2              |
| 1.9.6-1           | NVIDIA HPC SDK 20.3            |
| 1.9.6             | CUDA Toolkit 10.1 Update 2     |
| 1.9.5             | CUDA Toolkit 10.1 Update 1     |
| 1.9.4             | CUDA Toolkit 10.1              |
| 1.9.3             | CUDA Toolkit 10.0              |
| 1.9.2             | CUDA Toolkit 9.2               |
| 1.9.1             | CUDA Toolkit 9.1               |
| 1.9.0             | CUDA Toolkit 9.0               |
| 1.8.3             | CUDA Toolkit 8.0               |
| 1.8.2             | CUDA Toolkit 7.5               |
| 1.8.1             | CUDA Toolkit 7.0               |
| 1.8.0             |                                |
| 1.7.2             | CUDA Toolkit 6.5               |
| 1.7.1             | CUDA Toolkit 6.0               |
| 1.7.0             | CUDA Toolkit 5.5               |
| 1.6.0             |                                |
| 1.5.3             | CUDA Toolkit 5.0               |
| 1.5.2             | CUDA Toolkit 4.2               |
| 1.5.1             | CUDA Toolkit 4.1               |
| 1.5.0             |                                |
| 1.4.0             | CUDA Toolkit 4.0               |
| 1.3.0             | CUDA Toolkit 3.2               |
| 1.2.1             | CUDA Toolkit 3.1               |
| 1.2.0             |                                |
| 1.1.1             |                                |
| 1.1.0             |                                |
| 1.0.0             |                                |

## Branches and Tags

The following tag names are used in the Thrust project:

  * `github/nvhpc-X.Y`: the tag that directly corresponds to what has been shipped in the NVIDIA HPC SDK release X.Y.
  * `github/cuda-X.Y`: the tag that directly corresponds to what has been shipped in the CUDA Toolkit release X.Y.
  * `github/A.B.C`: the tag that directly corresponds to a Thrust version A.B.C.

The following branch names are used in the Thrust project:

  * `github/master`: the Source of Truth development branch of Thrust.
  * `github/old-master`: the old Source of Truth branch, before unification of public and internal repositories.
  * `github/feature/<name>`: feature branch for a feature under development.
  * `github/bug/<bug-system>/<bug-description>-<bug-id>`: bug fix branch, where `bug-system` is `github` or `nvidia`.
  * `gitlab/master`: mirror of `github/master`.
  * `perforce/private`: mirrored `github/master`, plus files necessary for internal NVIDIA testing systems.

On the rare occasion that we cannot do work in the open, for example when developing a change specific to an
unreleased product, these branches may exist on `gitlab` instead of `github`. By default, everything should be
in the open on `github` unless there is a strong motivation for it to not be open.

