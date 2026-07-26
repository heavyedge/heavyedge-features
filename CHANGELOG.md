# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## UNRELEASED

### Added

- API functions and command lines now take `n-chunks` argument.

### Changed

- API functions now yield the chunks instead of returning the full result.

### Fixed

- Segmented regression now does not get stuck.

## [1.1.0a1] - 2026-07-26

### Added

- Shape feature type is displayed on log from `shape-features` command.

## [1.1.0a0] - 2026-07-26

### Added

- `shape-features` command is added to replace `features-global` and `features-local`.

### Changed

- `heavyedge-landmarks` package is no longer required.
  Functions to locate landmarks are directly implemented.

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
