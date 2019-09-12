# Thrust Branching and Development Model

The following is a description of how the Thrust development teams approaches branching and release tagging. This
is a living document that will evolve as our process evolves.

## Thrust Version

Thrust has historically had its own versioning system, independent of the versioning scheme of the CUDA Toolkit.
Today, Thrust is released with the CUDA Toolkit, but we currently still maintain the double versioning scheme.

The following is a mapping from Thrust versions to CUDA Toolkit versions and vice versa. Note that some Thrust
versions don't directly map to any CUDA Toolkit version.

| Thrust version    | CUDA version  |
| ----------------- | ------------- |
| 1.9.6             | 10.1 Update 2 |
| 1.9.5             | 10.1 Update 1 |
| 1.9.4             | 10.1          |
| 1.9.3             | 10.0          |
| 1.9.2             | 9.2           |
| 1.9.1             | 9.1           |
| 1.9.0             | 9.0           |
| 1.8.3             | 8.0           |
| 1.8.2             | 7.5           |
| 1.8.1             | 7.0           |
| 1.8.0             | *N/A*         |
| 1.7.2             | 6.5           |
| 1.7.1             | 6.0           |
| 1.7.0             | 5.5           |
| 1.6.0             | *N/A*         |
| 1.5.3             | 5.0           |
| 1.5.2             | 4.2           |
| 1.5.1             | 4.1           |
| 1.5.0             | *N/A*         |
| 1.4.0             | 4.0           |
| 1.3.0             | 3.2           |
| 1.2.1             | 3.1           |
| 1.2.0             | *N/A*         |
| 1.1.1             | *N/A*         |
| 1.1.0             | *N/A*         |
| 1.0.0             | *N/A*         |

## Repositories

As Thrust is developed both on GitHub and internally at NVIDIA, there's three main places where code lives:

  * The [public Thrust repository](https://github.com/thrust/thrust), referred to as `github` later in this
    document.
  * An internal GitLab repository, referred to as `gitlab` later in this document.
  * An internal Perforce repository, referred to as `perforce` later in this document.

## Branches and Tags

The following tag names are used in the Thrust project:

  * `github/cuda-X.Y`: the tag that directly corresponds to what has been shipped in the CUDA Toolkit release X.Y.
  * `github/A.B.C`: the tag that directly corresponds to a Thrust version A.B.C.

The following branch names are used in the Thrust project:

  * `github/master`: the Source of Truth development branch of Thrust.
  * `github/old-master`: the old Source of Truth branch, before unification of public and internal repositories.
  * `perforce/private`: mirrored github/master, plus files necessary for internal NVIDIA testing systems.
  * `gitlab/staging/cuda-X.Y`: the branch for a CUDA Toolkit release that has not been released yet. cuda-X.Y should
    be tagged on this branch after the final commit freeze (see "Release branches" below).
  * `github/maintenance/cuda-Z.W`: the continuation of gitlab/staging/cuda-Z.W, but after release of CUDA Z.W, plus
    post-release fixes if any are needed (see "Old release branches" below).
  * `gitlab/feature/<name>`: feature branch for internally developed features.
  * `gitlab/bug/<bug-system>-<bug-id>`: bug fix branch, where `bug-system` is `github` or `nvbug`. Permits a description
    after `bug-id`.
  * `gitlab/master`: same as `github/master`, but not yet published, during a freezing period (see "Feature freeze"
    below).

## Development Process Described

### Normal development

During regular parts of the development cycle, when we develop features on feature branches, and fix bugs on the
main branch, we can:

  * Merge internal fixes to `github/master` and to `perforce/private`.
  * Merge Github contributions to `github/master` and to `perforce/private`.

### Feature freeze

In case where we have a new feature for a CUDA Toolkit release: just before the CUDA Toolkit feature freeze for a
new release branch, we should stop merging commits (including public contributions) to `github/master`, and move to
development on `gitlab/master`, and merge the not yet public features there.

In those cases, we should wait until the new version of the toolkit is released before we push the new updated
`gitlab/master` to `github/master`, roughly at the same time as we push from `gitlab/staging/cuda-X.Y` to
`github/maintenance/cuda-X.Y` and tag `cuda-X.Y`, and the appropriate Thrust version tag.

If we don't have big, not-public-before-release features landing in X.Y, however, we can avoid having a feature
freeze period.

The reason for having a freeze period at all is: `github/master` is supposed to be the Source of Truth. We want the
history to follow the same order of commits in both Git and Perforce, and once a change is merged, we cannot rebase
things that went into `perforce/internal` on top of it. Therefore: since we only really commit to Perforce but not
`github/master` when we have a feature that is ready to be delivered, but is only a part of a new release and
shouldn't/can't be public yet, we have to make sure that after it is merged to `gitlab/master` (and to `perforce/internal`),
nothing new lands in `github/master` before we push the feature out.

To avoid situations like this with bug fixes, when we fix a bug at a not crazy point in the release cycle, we
should develop it on git, merge/push it on Github, and then pull the new commit to Perforce.

### Release branches

These are the internal Git branches that map directly to internal CUDA release branches. These branches are primarily
developed in Git, and commits applied to them are then pushed to Perforce.

After a CUDA Toolkit version is released, these transition to being old release branches.

### Old release branches

These branches represent a version that has landed in a CUDA Toolkit version, but with bugfixes for things that do
deserve being fixed on a release branch. These shouldn't be groundbreaking; the following are an acceptable set of
fixes to go into these branches, because they can remove annoyances, but shouldn't change behavior:

  * Documentation fixes and updates.
  * Thrust build system changes.
  * Additional examples, fixes to examples and tests.
  * (Possibly:) Fixing missing headers. This one is slightly less obvious, because it makes it possible for users
    of standalone Thrust to write programs that won't compile with CUDA Thrust. Determinations will be made on a
    case by case basis.

