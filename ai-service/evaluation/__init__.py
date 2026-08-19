"""Evaluation harness for DeepVisionAttend.

Measures recognition accuracy against a labelled dataset using the production
pipeline, and renders the result as a report suitable for the project write-up.

* :mod:`evaluation.dataset` — loading and enrol/probe splitting.
* :mod:`evaluation.runner` — builds an in-memory gallery and scores probes.
* :mod:`evaluation.metrics` — verification and open-set identification metrics.
* :mod:`evaluation.figures` — SVG figures for the report.
* :mod:`evaluation.report` — console, Markdown, JSON and CSV output.
"""
