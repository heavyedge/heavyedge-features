# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-08-18

### Added

- `shape-features` command now quantifies `phi_i` values, which are signed information projection distance to each set of class probability whose argmax is the class `i`.

## [1.1.0] - 2026-08-17

### Added

- `shape-features` command is added to replace `features-global` and `features-local`.
- API functions and command lines now take `n-chunks` argument.
- Shape feature type is displayed on log from `shape-features` command.

### Changed

- I-projection is now computed using the closed form.
- API functions now yield the chunks instead of returning the full result.
- `heavyedge-landmarks` package is no longer required.
  Functions to locate landmarks are directly implemented.

### Removed

- `cvxpy` dependency is removed.

### Fixed

- Segmented regression now does not get stuck.

### Deprecated

- `features-global` command is deprecated. Use `shape-features` instead.
- `features-local` command is deprecated. Use `shape-features` instead.

## [1.0.1] - 2026-03-24

### Fixed

- Soft label input as csv file is now correctly handled.

## [1.0.0] - 2026-03-24

### Added

- Global shape features
    - `phi` : Signed information projection distance.

- Local shape features
    - `H` : Dimensionless edge height.
    - `b` : Edge width.
