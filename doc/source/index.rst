.. HeavyEdge-Features documentation master file, created by
   sphinx-quickstart on Sat Mar 21 21:59:10 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HeavyEdge-Features documentation
================================

HeavyEdge-Features is a Python package for quantifying shape features from coating edge profiles.

Usage
-----

HeavyEdge-Features can be used either as a command line program or as a Python module.

Command line
------------

Command line interface provides pre-defined subroutines for training and prediction.
It can be invoked by:

.. code-block:: bash

   heavyedge features-global <args>
   heavyedge features-local <args>

Refer to help message of each command for their arguments.

Python module
-------------

The Python module :mod:`heavyedge_features` provides functions and classes for Python runtime.
Refer to :ref:`api` section for high-level interface.

Module reference
================

.. module:: heavyedge_features

This section provides reference for :mod:`heavyedge_features` Python module.

.. _api:

Runtime API
-----------

.. automodule:: heavyedge_features.api
    :members:

Low-level API
-------------

.. automodule:: heavyedge_features.iproj
    :members:

.. automodule:: heavyedge_features.edge_width
    :members:
