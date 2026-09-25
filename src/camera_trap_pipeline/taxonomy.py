"""SpeciesNet-style label strings: parsing, roll-up to a confident taxon, and folding in events."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

UNRESOLVED = "animal (unresolved)"
LEVELS = ("class", "order", "family", "genus", "species")
_TAXONOMY_FIELDS = 5


@dataclass(frozen=True)
class Label:
    """A parsed classifier label.

    SpeciesNet labels look like ``<id>;<class>;<order>;<family>;<genus>;<species>;<common>``.
    Labels above species level leave the lower fields empty, and non-taxa such as
    ``blank`` leave every taxonomy field empty.

    Attributes:
        raw: The label exactly as the classifier returned it.
        taxonomy: Lower-cased class, order, family, genus and species fields.
        common_name: Lower-cased common name.
    """

    raw: str
    taxonomy: tuple[str, ...]
    common_name: str

    @property
    def depth(self) -> int:
        """How far down the taxonomy the label goes.

        Returns:
            Number of leading taxonomy fields that are filled in (0 for non-taxa).
        """
        depth = 0
        for part in self.taxonomy:
            if not part:
                break
            depth += 1
        return depth

    def lineage(self, depth: int | None = None) -> str:
        """Join the taxonomy fields down to a given depth.

        Args:
            depth: Number of fields to keep; defaults to all filled-in fields.

        Returns:
            The fields joined with ``;``.
        """
        keep = self.depth if depth is None else depth
        return ";".join(self.taxonomy[:keep])


def parse_label(raw: str) -> Label:
    """Parse a SpeciesNet-style label; any other string becomes a bare common name.

    Args:
        raw: Label string.

    Returns:
        The parsed label.
    """
    parts = raw.split(";")
    if len(parts) >= 2 + _TAXONOMY_FIELDS:
        taxonomy = tuple(p.strip().lower() for p in parts[1 : 1 + _TAXONOMY_FIELDS])
        common = parts[1 + _TAXONOMY_FIELDS].strip().lower()
        if not common:
            common = next((p for p in reversed(taxonomy) if p), "")
        return Label(raw=raw, taxonomy=taxonomy, common_name=common)
    return Label(raw=raw, taxonomy=("",) * _TAXONOMY_FIELDS, common_name=raw.strip().lower())


@dataclass(frozen=True)
class SpeciesCall:
    """The label assigned to one animal box after roll-up.

    Attributes:
        species: Display label: a common name, ``"<taxon> (<level>)"`` after roll-up, or
            ``animal (unresolved)``.
        score: Score of the label, or the summed score of the rolled-up group.
        level: ``species``, ``genus``, ``family``, ``order``, ``class`` or ``unresolved``.
        lineage: Taxonomy down to ``level`` joined with ``;``; empty when unresolved.
    """

    species: str
    score: float
    level: str
    lineage: str


def roll_up(classes: Sequence[str], scores: Sequence[float], threshold: float) -> SpeciesCall:
    """Pick the most specific label whose (summed) score reaches the threshold.

    The top label is kept when its own score is high enough. Otherwise the top-k scores are
    summed within each genus, then family, order and class, and the first group that
    reaches the threshold wins. For example, 0.40 white-tailed deer + 0.35 mule deer gives
    ``odocoileus (genus)`` at 0.75.

    Args:
        classes: Classifier labels, most likely first.
        scores: Scores matching ``classes``.
        threshold: Minimum score to accept a label.

    Returns:
        The chosen call; ``animal (unresolved)`` when nothing reaches the threshold or the
        winning label is not a taxon (for example ``blank``).
    """
    if not classes or not scores:
        return SpeciesCall(UNRESOLVED, 0.0, "unresolved", "")
    labels = [parse_label(c) for c in classes]
    top, top_score = labels[0], float(scores[0])
    if top_score >= threshold:
        if top.depth == 0:
            return SpeciesCall(UNRESOLVED, top_score, "unresolved", "")
        return SpeciesCall(top.common_name, top_score, LEVELS[top.depth - 1], top.lineage())
    for depth in range(_TAXONOMY_FIELDS - 1, 0, -1):
        totals: dict[str, float] = {}
        for label, score in zip(labels, scores, strict=False):
            if label.depth >= depth:
                key = label.lineage(depth)
                totals[key] = totals.get(key, 0.0) + float(score)
        if not totals:
            continue
        lineage, total = max(sorted(totals.items()), key=lambda item: item[1])
        if total >= threshold:
            return SpeciesCall(label_for_lineage(lineage), total, LEVELS[depth - 1], lineage)
    return SpeciesCall(UNRESOLVED, top_score, "unresolved", "")


def is_ancestor(lineage: str, other: str) -> bool:
    """Tell whether one lineage is a strict taxonomic ancestor of another.

    The empty lineage (unresolved) is an ancestor of every resolved lineage.

    Args:
        lineage: Candidate ancestor, ``;``-joined.
        other: Candidate descendant, ``;``-joined.

    Returns:
        True when ``lineage`` is a proper prefix of ``other``.
    """
    if lineage == other:
        return False
    return (lineage == "" and other != "") or other.startswith(lineage + ";")


def common_lineage(lineages: Iterable[str]) -> str:
    """Longest taxonomy prefix shared by every lineage (their closest common taxon).

    Args:
        lineages: ``;``-joined lineages; an empty string shares nothing.

    Returns:
        The shared prefix, or ``""`` when there is none.
    """
    split = [lineage.split(";") if lineage else [] for lineage in lineages]
    shared: list[str] = []
    for parts in zip(*split, strict=False):
        if len(set(parts)) != 1:
            break
        shared.append(parts[0])
    return ";".join(shared)


def label_for_lineage(lineage: str) -> str:
    """Name a taxon reached by roll-up.

    Args:
        lineage: ``;``-joined lineage.

    Returns:
        ``"<taxon> (<level>)"``, or ``animal (unresolved)`` for the empty lineage.
    """
    if not lineage:
        return UNRESOLVED
    parts = lineage.split(";")
    return f"{parts[-1]} ({LEVELS[len(parts) - 1]})"


def fold_labels(lineages: Mapping[str, str]) -> dict[str, str]:
    """Map vaguer labels in one event onto the single more specific label they contain.

    Within a burst the same animal is often labeled ``white-tailed deer`` in a clear frame
    and ``cervidae (family)`` in a blurry one. A vaguer label is folded into a more specific
    label only when exactly one label in the event descends from it, so two different
    species in one event are never merged.

    Args:
        lineages: Label to lineage for every label present in the event.

    Returns:
        Label to the label it should be counted as (itself when not folded).
    """
    leaves = [
        name
        for name, lineage in lineages.items()
        if not any(is_ancestor(lineage, other) for other in lineages.values())
    ]
    mapping: dict[str, str] = {}
    for name, lineage in lineages.items():
        descendants = [leaf for leaf in leaves if is_ancestor(lineage, lineages[leaf])]
        mapping[name] = descendants[0] if len(descendants) == 1 else name
    return mapping
