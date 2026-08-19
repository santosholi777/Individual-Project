"""Loading and splitting an evaluation dataset.

Expected layout — one directory per identity, named by student id::

    dataset/
    ├── CS2021001/
    │   ├── enrol/               # optional: explicit enrolment images
    │   │   ├── 01.jpg
    │   │   └── 02.jpg
    │   ├── frontal/             # optional: condition sub-folders
    │   │   └── 01.jpg
    │   ├── dim_light/
    │   │   └── 01.jpg
    │   └── 05.jpg               # or just loose files
    └── CS2021002/
        └── ...

Two conventions, both optional:

* An ``enrol/`` sub-folder marks the enrolment images explicitly. Without one,
  the first ``--enrol-per-identity`` images (sorted by name) are used.
* Any other sub-folder names the **condition** of the probes inside it
  (``dim_light``, ``side_angle``, ``mask``, …). Loose files are tagged
  ``default``. Conditions are what let the report answer "where does it break?"
  rather than just "how accurate is it?".

An optional ``meta.json`` at the dataset root supplies display names and, for
fairness reporting, a group label per identity::

    {
      "CS2021001": {"name": "Aditi Sharma", "group": "female"},
      "CS2021002": {"name": "Rahul Verma", "group": "male"}
    }

Group labels are never inferred — only what the evaluator explicitly provides is
reported on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from exceptions import DeepVisionAttendError
from logging_config import get_logger
from utils.image_utils import SUPPORTED_EXTENSIONS

logger = get_logger(__name__)

#: Sub-folder name that explicitly marks enrolment images.
ENROL_DIR = "enrol"
#: Condition assigned to images that sit directly in an identity's folder.
DEFAULT_CONDITION = "default"


class DatasetError(DeepVisionAttendError):
    """The evaluation dataset is missing, empty or malformed."""

    error_code = "dataset_error"


@dataclass(slots=True)
class ProbeImage:
    """One image to be recognised during evaluation."""

    path: Path
    #: The identity this image really belongs to (ground truth).
    identity: str
    #: Capture condition, from the sub-folder name.
    condition: str = DEFAULT_CONDITION


@dataclass(slots=True)
class Identity:
    """One person in the dataset."""

    student_id: str
    name: str
    enrol_paths: list[Path] = field(default_factory=list)
    probes: list[ProbeImage] = field(default_factory=list)
    #: Optional demographic/appearance group, for fairness reporting only.
    group: str | None = None

    @property
    def usable(self) -> bool:
        """Whether this identity can contribute to the evaluation at all."""
        return bool(self.enrol_paths) and bool(self.probes)


@dataclass(slots=True)
class Dataset:
    """A loaded evaluation dataset.

    Attributes:
        identities: People enrolled into the gallery and probed.
        unknowns: People deliberately *not* enrolled. Their probes must all be
            rejected — this is the direct test of proxy-attendance resistance.
        root: Where the dataset was loaded from.
    """

    identities: list[Identity] = field(default_factory=list)
    unknowns: list[Identity] = field(default_factory=list)
    root: Path = Path()

    @property
    def probe_count(self) -> int:
        """Total probes across enrolled identities."""
        return sum(len(identity.probes) for identity in self.identities)

    @property
    def unknown_probe_count(self) -> int:
        """Total probes from unenrolled (stranger) identities."""
        return sum(len(identity.probes) for identity in self.unknowns)

    @property
    def enrol_count(self) -> int:
        """Total enrolment images."""
        return sum(len(identity.enrol_paths) for identity in self.identities)

    @property
    def conditions(self) -> list[str]:
        """Every condition label present among the probes, sorted."""
        labels = {
            probe.condition
            for identity in self.identities
            for probe in identity.probes
        }
        return sorted(labels)

    @property
    def groups(self) -> list[str]:
        """Every group label supplied in meta.json, sorted."""
        labels = {
            identity.group for identity in self.identities if identity.group
        }
        return sorted(labels)

    def summary(self) -> dict[str, object]:
        """A short description of what was loaded, for the report header."""
        return {
            "root": str(self.root),
            "identities": len(self.identities),
            "enrol_images": self.enrol_count,
            "probe_images": self.probe_count,
            "unknown_identities": len(self.unknowns),
            "unknown_probe_images": self.unknown_probe_count,
            "conditions": self.conditions,
            "groups": self.groups,
        }


def _image_files(directory: Path) -> list[Path]:
    """Return supported images directly inside ``directory``, sorted by name."""
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _load_meta(root: Path) -> dict[str, dict[str, str]]:
    """Read the optional meta.json, tolerating its absence.

    Raises:
        DatasetError: If the file exists but is not readable JSON.
    """
    meta_path = root / "meta.json"
    if not meta_path.is_file():
        return {}

    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(
            f"Could not read {meta_path}: {exc}", details={"path": str(meta_path)}
        ) from exc

    if not isinstance(raw, dict):
        raise DatasetError("meta.json must be an object keyed by student id")
    return raw


def load_dataset(
    root: str | Path,
    enrol_per_identity: int = 3,
    holdout: tuple[str, ...] = (),
) -> Dataset:
    """Load an evaluation dataset from disk.

    Args:
        root: Dataset directory.
        enrol_per_identity: How many images to enrol per identity when there is
            no explicit ``enrol/`` sub-folder. The rest become probes.
        holdout: Student ids to treat as strangers — excluded from the gallery,
            with every one of their images used as an unknown probe.

    Returns:
        The loaded dataset.

    Raises:
        DatasetError: If the directory is missing, or holds no usable identity.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise DatasetError(
            f"Dataset directory not found: {root_path}",
            details={"path": str(root_path)},
        )

    meta = _load_meta(root_path)
    dataset = Dataset(root=root_path)
    holdout_set = set(holdout)
    skipped: list[str] = []

    for identity_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
        student_id = identity_dir.name
        entry = meta.get(student_id, {})
        identity = Identity(
            student_id=student_id,
            name=str(entry.get("name", student_id)),
            group=entry.get("group"),
        )

        loose = _image_files(identity_dir)
        explicit_enrol_dir = identity_dir / ENROL_DIR
        condition_dirs = [
            path
            for path in sorted(identity_dir.iterdir())
            if path.is_dir() and path.name != ENROL_DIR
        ]

        # An identity held out is a stranger: everything they have is a probe,
        # because they must never be enrolled.
        if student_id in holdout_set:
            for path in loose:
                identity.probes.append(
                    ProbeImage(path=path, identity=student_id, condition=DEFAULT_CONDITION)
                )
            if explicit_enrol_dir.is_dir():
                for path in _image_files(explicit_enrol_dir):
                    identity.probes.append(
                        ProbeImage(path=path, identity=student_id, condition=DEFAULT_CONDITION)
                    )
            for condition_dir in condition_dirs:
                for path in _image_files(condition_dir):
                    identity.probes.append(
                        ProbeImage(
                            path=path, identity=student_id, condition=condition_dir.name
                        )
                    )
            if identity.probes:
                dataset.unknowns.append(identity)
            else:
                skipped.append(f"{student_id} (held out but has no images)")
            continue

        if explicit_enrol_dir.is_dir():
            identity.enrol_paths = _image_files(explicit_enrol_dir)
            probe_paths = loose
        else:
            # Split the loose files: the first N enrol, the remainder probe.
            identity.enrol_paths = loose[:enrol_per_identity]
            probe_paths = loose[enrol_per_identity:]

        for path in probe_paths:
            identity.probes.append(
                ProbeImage(path=path, identity=student_id, condition=DEFAULT_CONDITION)
            )
        for condition_dir in condition_dirs:
            for path in _image_files(condition_dir):
                identity.probes.append(
                    ProbeImage(
                        path=path, identity=student_id, condition=condition_dir.name
                    )
                )

        if identity.usable:
            dataset.identities.append(identity)
        else:
            reason = (
                "no enrolment images"
                if not identity.enrol_paths
                else "no probe images left after the enrolment split"
            )
            skipped.append(f"{student_id} ({reason})")

    for note in skipped:
        logger.warning("Skipping identity: %s", note)

    if not dataset.identities:
        raise DatasetError(
            "No usable identities found. Each identity needs at least one "
            "enrolment image and one probe image. With "
            f"--enrol-per-identity {enrol_per_identity}, that means at least "
            f"{enrol_per_identity + 1} images per person.",
            details={"path": str(root_path), "skipped": skipped},
        )

    logger.info(
        "Loaded %s identities (%s enrol, %s probes) and %s unknown identities "
        "(%s probes) from %s",
        len(dataset.identities),
        dataset.enrol_count,
        dataset.probe_count,
        len(dataset.unknowns),
        dataset.unknown_probe_count,
        root_path,
    )
    return dataset
