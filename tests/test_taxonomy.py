from __future__ import annotations

import pytest

from camera_trap_pipeline.taxonomy import (
    UNRESOLVED,
    common_lineage,
    fold_labels,
    is_ancestor,
    label_for_lineage,
    parse_label,
    roll_up,
)

DEER = "id1;mammalia;cetartiodactyla;cervidae;odocoileus;virginianus;white-tailed deer"
MULE = "id2;mammalia;cetartiodactyla;cervidae;odocoileus;hemionus;mule deer"
ELK = "id3;mammalia;cetartiodactyla;cervidae;cervus;canadensis;elk"
COYOTE = "id4;mammalia;carnivora;canidae;canis;latrans;coyote"
MAMMAL = "id5;mammalia;;;;;mammal"
BLANK = "id6;;;;;;blank"


def test_parse_label_reads_taxonomy_and_common_name() -> None:
    label = parse_label(DEER)
    assert label.taxonomy == (
        "mammalia",
        "cetartiodactyla",
        "cervidae",
        "odocoileus",
        "virginianus",
    )
    assert label.common_name == "white-tailed deer"
    assert label.depth == 5
    assert label.lineage(3) == "mammalia;cetartiodactyla;cervidae"
    assert parse_label(MAMMAL).depth == 1
    assert parse_label(BLANK).depth == 0
    assert parse_label("id;aves;;;;;").common_name == "aves"
    assert parse_label("Coyote").common_name == "coyote"
    assert parse_label("Coyote").depth == 0


def test_confident_top_label_is_kept() -> None:
    call = roll_up([DEER, MULE], [0.9, 0.05], threshold=0.65)
    assert (call.species, call.level) == ("white-tailed deer", "species")
    assert call.lineage.endswith("odocoileus;virginianus")


def test_confident_higher_taxon_label_keeps_its_level() -> None:
    call = roll_up([MAMMAL, DEER], [0.8, 0.1], threshold=0.65)
    assert (call.species, call.level, call.lineage) == ("mammal", "class", "mammalia")


def test_uncertain_label_rolls_up_to_genus_then_family() -> None:
    genus = roll_up([DEER, MULE, COYOTE], [0.40, 0.35, 0.10], threshold=0.65)
    assert (genus.species, genus.level) == ("odocoileus (genus)", "genus")
    assert genus.score == pytest.approx(0.75)

    family = roll_up([DEER, ELK, COYOTE], [0.40, 0.30, 0.20], threshold=0.65)
    assert (family.species, family.level) == ("cervidae (family)", "family")

    mammal = roll_up([DEER, COYOTE], [0.40, 0.30], threshold=0.65)
    assert (mammal.species, mammal.level) == ("mammalia (class)", "class")


def test_nothing_confident_or_non_taxon_is_unresolved() -> None:
    assert roll_up([DEER, COYOTE], [0.3, 0.2], threshold=0.65).species == UNRESOLVED
    assert roll_up([BLANK], [0.95], threshold=0.65).species == UNRESOLVED
    assert roll_up([], [], threshold=0.65).level == "unresolved"


def test_is_ancestor() -> None:
    assert is_ancestor("", "mammalia")
    assert is_ancestor("mammalia", "mammalia;carnivora")
    assert not is_ancestor("mammalia", "mammalia")
    assert not is_ancestor("mammalia;carn", "mammalia;carnivora")
    assert not is_ancestor("", "")


def test_fold_labels_merges_vague_labels_into_single_specific_one() -> None:
    lineages = {
        "white-tailed deer": "mammalia;cetartiodactyla;cervidae;odocoileus;virginianus",
        "cervidae (family)": "mammalia;cetartiodactyla;cervidae",
        UNRESOLVED: "",
    }
    assert fold_labels(lineages) == {
        "white-tailed deer": "white-tailed deer",
        "cervidae (family)": "white-tailed deer",
        UNRESOLVED: "white-tailed deer",
    }


def test_fold_labels_keeps_two_species_apart() -> None:
    lineages = {
        "white-tailed deer": "mammalia;cetartiodactyla;cervidae;odocoileus;virginianus",
        "coyote": "mammalia;carnivora;canidae;canis;latrans",
        "mammalia (class)": "mammalia",
    }
    mapping = fold_labels(lineages)
    assert mapping["white-tailed deer"] == "white-tailed deer"
    assert mapping["coyote"] == "coyote"
    assert mapping["mammalia (class)"] == "mammalia (class)"


def test_common_lineage_and_its_label() -> None:
    deer = "mammalia;cetartiodactyla;cervidae;odocoileus;virginianus"
    elk = "mammalia;cetartiodactyla;cervidae;cervus;canadensis"
    assert common_lineage([deer, elk]) == "mammalia;cetartiodactyla;cervidae"
    assert common_lineage([deer, "aves;galliformes"]) == ""
    assert common_lineage([deer, ""]) == ""
    assert label_for_lineage("mammalia;cetartiodactyla;cervidae") == "cervidae (family)"
    assert label_for_lineage("") == UNRESOLVED
