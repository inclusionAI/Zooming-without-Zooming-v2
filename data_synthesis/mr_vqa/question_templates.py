import random
from collections import defaultdict


def _default_answer_kind(question_type):
    if question_type in {
        "aggregate_count", "aggregate_attribute_count", "aggregate_numeric",
        "arithmetic_composition", "conditional_composition",
    }:
        return "numeric"
    if question_type == "boolean_composition":
        return "boolean"
    if question_type == "cross_region_pairing":
        return "semantic_pair"
    if question_type in {
        "set_filtering", "anomaly_detection", "extremum",
        "candidate_matching", "spatial_chain", "geometric_line_relation",
        "topological_relation", "depth_ordering",
    }:
        return "anchor"
    return "semantic"


def _t(template_id, question_type, min_regions, max_regions, instruction,
       necessity_mode=None, answer_semantics="scalar", final_answer_kind=None,
       evidence_scope="any"):
    return {
        "template_id": template_id,
        "question_type": question_type,
        "min_regions": min_regions,
        "max_regions": max_regions,
        "necessity_mode": necessity_mode,
        "answer_semantics": answer_semantics,
        "final_answer_kind": final_answer_kind or _default_answer_kind(question_type),
        "evidence_scope": evidence_scope,
        "instruction": instruction,
    }


# Templates describe reasoning programs, not visual facts. The designer model must
# skip a template when the current crops do not objectively support it.
QUESTION_TEMPLATES = [
    # Aggregation
    _t("count.same_class.total", "aggregate_count", 3, 7,
       "Count the same concrete object class in every region and sum the positive local counts."),
    _t("count.parts.total", "aggregate_count", 3, 6,
       "Count repeated visible parts of the same kind across all regions."),
    _t("count.grouped_instances", "aggregate_count", 3, 7,
       "Each region contains a small group of the same target; add all group sizes."),
    _t("count.multi_instance_regions", "aggregate_count", 3, 6,
       "Use regions with unequal positive counts so the answer is not the number of boxes."),
    _t("attr_count.color", "aggregate_attribute_count", 3, 7,
       "Count same-class instances having one objective color attribute in every region."),
    _t("attr_count.state", "aggregate_attribute_count", 3, 7,
       "Count same-class instances sharing one clearly visible state or accessory."),
    _t("attr_count.shape", "aggregate_attribute_count", 3, 7,
       "Count repeated elements with one objective shape or structural feature."),
    _t("attr_count.text_marker", "aggregate_attribute_count", 3, 6,
       "Count items carrying the same visible symbol, digit, or text marker."),
    _t("numeric.sum_intrinsic", "aggregate_numeric", 3, 6,
       "Read one intrinsic scene number from every region and add them."),
    _t("numeric.sequence_total", "aggregate_numeric", 3, 6,
       "Read a short numeric sequence distributed across regions and compute its total."),
    _t("numeric.price_quantity", "aggregate_numeric", 3, 5,
       "Combine visible prices, quantities, scores, or dates only when the scene makes the operation meaningful."),

    # Closed-set selection and comparison
    _t("filter.single_attribute", "set_filtering", 3, 7,
       "Return the complete subset of comparable candidates satisfying one objective attribute.",
       answer_semantics="unordered_set"),
    _t("filter.attribute_intersection", "set_filtering", 3, 7,
       "Return candidates satisfying two simultaneous visible attributes.",
       answer_semantics="unordered_set"),
    _t("filter.attribute_exclusion", "set_filtering", 3, 7,
       "Return candidates having attribute A but not attribute B.",
       answer_semantics="unordered_set"),
    _t("filter.part_presence", "set_filtering", 3, 7,
       "Return all comparable objects containing a specified visible part.",
       answer_semantics="unordered_set"),
    _t("filter.text_or_symbol", "set_filtering", 3, 7,
       "Return all candidates containing a specified text, digit, logo, or symbol property.",
       answer_semantics="unordered_set"),
    _t("filter.relation", "set_filtering", 3, 7,
       "Return candidates that stand in the same objective spatial or part-whole relation.",
       answer_semantics="unordered_set"),
    _t("filter.then_object_labels", "set_filtering", 3, 7,
       "Filter the complete candidate set by an objective property, then return natural object/category labels for every qualifying member instead of outline colors.",
       answer_semantics="unordered_set", final_answer_kind="semantic_set"),
    _t("filter.then_attributes", "set_filtering", 3, 7,
       "Filter the complete candidate set, then return one objective visible attribute from each qualifying object as a semantic set rather than outline colors.",
       answer_semantics="unordered_set", final_answer_kind="semantic_set"),
    _t("anomaly.attribute", "anomaly_detection", 3, 7,
       "Find the unique candidate differing in one objective attribute."),
    _t("anomaly.missing_part", "anomaly_detection", 3, 7,
       "Find the only candidate missing a repeated visible part."),
    _t("anomaly.count", "anomaly_detection", 3, 7,
       "Find the only candidate with a different local instance count."),
    _t("anomaly.text_pattern", "anomaly_detection", 3, 7,
       "Find the only text or number region that breaks a clear pattern."),
    _t("anomaly.then_attribute", "anomaly_detection", 3, 7,
       "Find the unique anomaly using all candidates, then report a different objective visible attribute of that selected object rather than its outline color.",
       final_answer_kind="semantic"),
    _t("anomaly.then_object", "anomaly_detection", 3, 7,
       "Find the unique anomaly using one objective comparison, then answer with the selected object's natural category or role, never its outline color.",
       final_answer_kind="semantic"),
    _t("anomaly.then_text", "anomaly_detection", 3, 7,
       "Find the unique visual anomaly, then read a short word, number, or symbol from that selected region.",
       final_answer_kind="text"),
    _t("extremum.count", "extremum", 3, 7,
       "Select the candidate with the largest or smallest local count."),
    _t("extremum.visible_number", "extremum", 3, 7,
       "Select the largest or smallest clearly readable number."),
    _t("extremum.rank_second", "extremum", 4, 7,
       "Rank comparable values and select the second-largest or second-smallest candidate."),
    _t("extremum.then_attribute", "extremum", 3, 7,
       "Select the objective count or numeric extremum across all candidates, then report another visible attribute of the selected object.",
       final_answer_kind="semantic"),
    _t("extremum.then_object", "extremum", 3, 7,
       "Select the objective extremum across all candidates, then answer with the selected object's natural category or role instead of an outline color.",
       final_answer_kind="semantic"),
    _t("extremum.then_text", "extremum", 3, 7,
       "Select the extremum using counts or clearly readable numbers, then return text or a symbol from the selected region.",
       final_answer_kind="text"),

    # Matching and OCR
    _t("match.reference_attribute", "candidate_matching", 3, 7,
       "Use one reference region's attribute to select its unique match from a closed candidate set."),
    _t("match.symbol_object", "candidate_matching", 3, 7,
       "Match a reference symbol or icon to the corresponding object candidate."),
    _t("match.text_initial", "candidate_matching", 3, 7,
       "Match candidates by a visible initial letter, digit, or short text pattern."),
    _t("match.part_whole", "candidate_matching", 3, 7,
       "Match a visible part to the correct whole object among candidates."),
    _t("match.same_pattern", "candidate_matching", 3, 7,
       "Match a reference color-shape or texture pattern to one candidate."),
    _t("match.then_attribute", "candidate_matching", 3, 7,
       "Resolve a unique reference-to-candidate match, then report a second visible attribute of the matched object rather than its outline color.",
       final_answer_kind="semantic"),
    _t("match.then_object", "candidate_matching", 3, 7,
       "Resolve a unique match, then answer with the matched object's natural category, role, or action.",
       final_answer_kind="semantic"),
    _t("match.then_text", "candidate_matching", 3, 7,
       "Resolve a unique visual match, then read a short text, number, or symbol from the matched region.",
       final_answer_kind="text"),
    _t("pair.person_object", "cross_region_pairing", 4, 7,
       "Produce objective person-object pairs using clothing, possession, or adjacency cues.",
       answer_semantics="ordered_pair"),
    _t("pair.object_text", "cross_region_pairing", 4, 7,
       "Pair objects with their visible labels, prices, signs, or numbers.",
       answer_semantics="ordered_pair"),
    _t("pair.part_whole", "cross_region_pairing", 4, 7,
       "Pair separated fine-part crops with their corresponding whole objects.",
       answer_semantics="ordered_pair"),
    _t("pair.shared_attribute", "cross_region_pairing", 4, 7,
       "Pair candidates across two sets by one unique objective attribute.",
       answer_semantics="ordered_pair"),
    _t("ocr.text_to_object", "ocr_visual_verification", 3, 6,
       "Read a reference word or number, match it to visual candidates, then return the matched object."),
    _t("ocr.label_to_attribute", "ocr_visual_verification", 3, 6,
       "Read a label, locate its matching object, and report one visible attribute of that object."),
    _t("ocr.sequence_reconstruct", "ocr_visual_verification", 3, 7,
       "Read ordered text fragments from all regions and reconstruct the complete phrase."),
    _t("ocr.number_to_candidate", "ocr_visual_verification", 3, 6,
       "Read a number in one region and use it to select or verify a candidate in other regions."),

    # Spatial, relational, and dependent chains
    _t("spatial.between", "spatial_chain", 3, 5,
       "Use two anchor regions to identify the target located between them."),
    _t("spatial.two_constraints", "spatial_chain", 3, 5,
       "Identify a target satisfying two independent spatial constraints from different anchors."),
    _t("spatial.direction_chain", "spatial_chain", 3, 6,
       "Follow a left/right/above/below chain through several regions to the final target."),
    _t("spatial.nearest_reference", "spatial_chain", 3, 6,
       "Use a reference and all candidates to determine the uniquely nearest or farthest marked target."),
    _t("spatial.then_attribute", "spatial_chain", 3, 6,
       "Use a multi-step spatial chain to identify the final target, then report one objective visible attribute of it rather than its outline color.",
       final_answer_kind="semantic"),
    _t("spatial.then_object", "spatial_chain", 3, 6,
       "Follow spatial constraints through all regions, then answer with the final target's natural object category or role.",
       final_answer_kind="semantic"),
    _t("viewpoint.person_left_right", "viewpoint_transform", 3, 5,
       "Use a clearly front- or back-facing person to establish the person's coordinate frame, transform two target positions into that frame, and answer their left/right relation.",
       final_answer_kind="semantic"),
    _t("viewpoint.opposite_side", "viewpoint_transform", 3, 5,
       "For an unambiguous planar arrangement, reason how the left/right or near/far relation changes when viewed from the explicitly opposite side.",
       final_answer_kind="semantic"),
    _t("viewpoint.object_frame", "viewpoint_transform", 3, 5,
       "Use the visible front or heading of an oriented object as the local frame, then transform target positions into that object's left/right/front/back directions.",
       final_answer_kind="semantic"),
    _t("agent.person_relative_side", "agent_relative_direction", 3, 5,
       "Determine which target lies to a clearly oriented person's left or right; do not substitute camera-left for person-left.",
       final_answer_kind="semantic"),
    _t("agent.vehicle_heading", "agent_relative_direction", 3, 5,
       "Use a vehicle, boat, or bicycle heading as forward and determine a target's left/right/front/behind relation in that local frame.",
       final_answer_kind="semantic"),
    _t("agent.select_then_report", "agent_relative_direction", 3, 6,
       "Use an agent-relative direction to select one target, then report a second objective visible property of it.",
       final_answer_kind="semantic"),
    _t("geometry.classify_pair", "geometric_line_relation", 3, 6,
       "Compare clearly visible dominant edges and identify the pair whose apparent image-plane relation is parallel, perpendicular, or intersecting.",
       answer_semantics="ordered_pair", final_answer_kind="semantic_pair"),
    _t("geometry.unique_intersection", "geometric_line_relation", 3, 7,
       "Treat all marked line or path candidates as a closed set and find the unique pair that intersects in the image plane.",
       answer_semantics="ordered_pair", final_answer_kind="semantic_pair"),
    _t("geometry.parallel_family", "geometric_line_relation", 3, 7,
       "Return the complete subset of marked edges belonging to one clearly supported parallel family.",
       answer_semantics="unordered_set"),
    _t("geometry.perspective_parallel", "geometric_line_relation", 3, 6,
       "Only with a clear vanishing-point or repeated architectural cue, identify physical 3D parallels that converge under perspective.",
       answer_semantics="unordered_set"),
    _t("topology.touching_connected", "topological_relation", 3, 7,
       "Return the complete subset of marked structures that touch or connect to one explicit reference structure.",
       answer_semantics="unordered_set"),
    _t("topology.containment", "topological_relation", 3, 7,
       "Compare all marked candidates and identify the one contained by, surrounding, or overlapping the reference boundary.",
       final_answer_kind="semantic"),
    _t("topology.path_connectivity", "topological_relation", 3, 7,
       "Use all marked road, path, rail, pipe, or branch segments to determine the complete connected route or the disconnected segment.",
       answer_semantics="unordered_set"),
    _t("depth.global_nearest", "depth_ordering", 3, 7,
       "Compare all candidates using unambiguous occlusion, ground contact, and perspective cues to select the nearest or farthest object.",
       final_answer_kind="semantic"),
    _t("depth.occlusion_order", "depth_ordering", 3, 6,
       "Use mutual occlusion evidence to identify which marked object is in front of another, with every region needed to resolve identity and depth.",
       final_answer_kind="semantic"),
    _t("depth.rank_then_attribute", "depth_ordering", 3, 7,
       "Rank a closed candidate set by clear depth cues, select a requested rank, then report a visible attribute of that object.",
       final_answer_kind="semantic"),
    _t("transitivity.horizontal", "spatial_transitivity", 3, 6,
       "Compose two or more left/right relations through shared entities to derive a relation not stated by any single region.",
       final_answer_kind="semantic"),
    _t("transitivity.vertical", "spatial_transitivity", 3, 6,
       "Compose above/below relations through intermediate objects to derive the final vertical relation.",
       final_answer_kind="semantic"),
    _t("transitivity.select_then_report", "spatial_transitivity", 3, 7,
       "Use a transitive spatial chain to select the final object, then report a second visible property from the last region.",
       final_answer_kind="semantic"),
    _t("dependent.identity_property", "dependent_chain", 3, 6,
       "Identify an entity from early clues, then report its property from a later region."),
    _t("dependent.reference_match_lookup", "dependent_chain", 3, 6,
       "Derive a reference attribute, match a candidate, then look up a second attribute."),
    _t("dependent.part_whole_property", "dependent_chain", 3, 6,
       "Identify a part, resolve its whole object, then answer about another visible part or property."),
    _t("dependent.order_lookup", "dependent_chain", 3, 6,
       "Derive an order or index from early regions and use it to read a later candidate."),
    _t("relation.material_transfer", "relational_composition", 3, 6,
       "Compare reference materials or structures, then classify a target by the matching relation."),
    _t("relation.part_whole", "relational_composition", 3, 6,
       "Combine part-whole evidence from multiple regions into one final object or property answer."),
    _t("relation.transitive_spatial", "relational_composition", 3, 6,
       "Compose two objective spatial relations to derive a third relation."),
    _t("relation.shared_owner", "relational_composition", 3, 6,
       "Use multiple possession or attachment cues to identify a shared owner or object."),
    _t("relation.attribute_analogy", "relational_composition", 4, 7,
       "Use an A:B visible attribute relation to choose the candidate completing C:?",
       answer_semantics="ordered_pair"),

    # Programmatic composition. These are deliberately synthetic but verifiable.
    _t("arithmetic.heterogeneous_sum", "arithmetic_composition", 3, 5,
       "Case-17 style: add independent heterogeneous values such as two people counts and one word count."),
    _t("arithmetic.multiply_add", "arithmetic_composition", 3, 5,
       "Multiply values from two regions, then add a value from another region."),
    _t("arithmetic.sum_difference", "arithmetic_composition", 3, 6,
       "Add values from one subset of regions and subtract values from another subset."),
    _t("arithmetic.ocr_count", "arithmetic_composition", 3, 5,
       "Combine a clearly read OCR number with one or more visual object counts."),
    _t("arithmetic.count_then_scale", "arithmetic_composition", 3, 5,
       "Sum two local counts and multiply the result by a third visible value."),
    _t("arithmetic.absolute_difference", "arithmetic_composition", 3, 6,
       "Compute two multi-region subtotals and return their nonnegative difference."),
    _t("conditional.shared_payload", "conditional_composition", 3, 6,
       "Use early regions to choose add versus subtract, while both branches use the same remaining payload regions."),
    _t("conditional.compare_then_formula", "conditional_composition", 4, 7,
       "Compare two visible values to select a formula that still consumes every remaining region."),
    _t("conditional.text_then_lookup", "conditional_composition", 3, 6,
       "Evaluate a text equality condition, then apply the chosen lookup rule to shared candidate evidence."),
    _t("boolean.two_attributes", "boolean_composition", 3, 6,
       "Evaluate two objective attribute predicates and return the four-state truth pattern."),
    _t("boolean.count_and_relation", "boolean_composition", 3, 6,
       "Evaluate one count comparison and one spatial or attribute relation."),
    _t("boolean.two_comparisons", "boolean_composition", 3, 6,
       "Evaluate two independent ordered comparisons and classify their truth pattern."),
    _t("boolean.ocr_and_visual", "boolean_composition", 3, 6,
       "Evaluate one OCR predicate and one visual predicate, then return the four-state pattern."),
    _t("multihop.ocr_match_attribute", "multi_hop_reasoning", 3, 6,
       "Read text, match a candidate, then report a visible attribute of the matched candidate."),
    _t("multihop.count_index_select", "multi_hop_reasoning", 3, 7,
       "Compute a count from early regions and use it as an index into an ordered candidate set."),
    _t("multihop.filter_then_extremum", "multi_hop_reasoning", 4, 7,
       "Filter candidates by one attribute, then select an extremum within the filtered subset."),
    _t("multihop.reference_pair_lookup", "multi_hop_reasoning", 4, 7,
       "Resolve a reference, form the correct pair, then look up the pair's final property."),
    _t("multihop.relation_chain", "multi_hop_reasoning", 3, 7,
       "Follow a 2-5 hop identity, spatial, or part-whole relation chain."),
    _t("multihop.compute_match_read", "multi_hop_reasoning", 4, 7,
       "Compute a value, match it to a candidate label or count, then read the selected target's attribute."),
    _t("multihop.select_then_attribute", "multi_hop_reasoning", 3, 7,
       "Use all regions to select one target by comparison, matching, or indexing, then report a second objective attribute of that target.",
       final_answer_kind="semantic"),
    _t("multihop.select_then_action", "multi_hop_reasoning", 3, 7,
       "Identify a person or object through a multi-region chain, then report its visible action, role, or state.",
       final_answer_kind="semantic"),
    _t("multihop.select_then_text", "multi_hop_reasoning", 3, 7,
       "Identify the correct region through at least two visual hops, then return its short visible text, digit, or symbol.",
       final_answer_kind="text"),
    _t("cross_image.correspondence", "cross_image_composition", 3, 7,
       "Find a shared object, part, symbol, texture, or attribute across 2-5 labeled source images, then report the unique correspondence.",
       final_answer_kind="semantic", evidence_scope="cross_image"),
    _t("cross_image.compare_then_report", "cross_image_composition", 3, 7,
       "Compare evidence distributed across source images to select one image or region, then report a second objective property from it.",
       final_answer_kind="semantic", evidence_scope="cross_image"),
    _t("cross_image.multi_hop", "cross_image_composition", 3, 7,
       "Resolve a clue in one image, match it in another image, and use one or more additional images to obtain the final short answer.",
       final_answer_kind="semantic", evidence_scope="cross_image"),
    _t("cross_image.relation_comparison", "cross_image_composition", 4, 7,
       "Determine an objective within-image relation in each source image, compare those relations, and return the unique matching or differing case.",
       final_answer_kind="semantic", evidence_scope="cross_image"),
    _t("cross_image.text_visual_chain", "cross_image_composition", 3, 7,
       "Read text or a symbol in one source image, use it to select visual evidence in other images, then return a final visible property.",
       final_answer_kind="text", evidence_scope="cross_image"),
]


def templates_by_type():
    grouped = defaultdict(list)
    for spec in QUESTION_TEMPLATES:
        grouped[spec["question_type"]].append(spec)
    return dict(grouped)


def select_question_templates(num_regions, count, seed, preferred_type=None,
                              evidence_scope="single_image"):
    """Select compatible templates with output-form and type diversity."""
    compatible = [
        spec for spec in QUESTION_TEMPLATES
        if (spec["min_regions"] <= num_regions <= spec["max_regions"]
            and spec.get("evidence_scope", "any") in {"any", evidence_scope})
    ]
    rng = random.Random(seed)
    rng.shuffle(compatible)
    rich_kinds = {"semantic", "text", "semantic_pair", "semantic_set"}
    target_schedule = [
        "semantic", "text", "anchor", "semantic_pair", "boolean", "numeric",
        "anchor", "semantic",
    ]
    target_schedule.extend("semantic" for _ in range(max(0, count - len(target_schedule))))
    selected = []
    used_types = set()
    # Accurate, unambiguous localization answers are useful. Keep them as a
    # substantial minority instead of suppressing them to a token slot.
    max_anchor = max(1, (count * 3 + 7) // 8)
    max_numeric = max(1, count // 4)

    def allowed(spec):
        kind = spec["final_answer_kind"]
        if kind == "anchor" and sum(x["final_answer_kind"] == "anchor" for x in selected) >= max_anchor:
            return False
        if kind == "numeric" and sum(x["final_answer_kind"] == "numeric" for x in selected) >= max_numeric:
            return False
        return True

    while len(selected) < count:
        slot = len(selected)
        target_kind = target_schedule[slot]
        pool = [spec for spec in compatible if spec not in selected and allowed(spec)]
        if not pool:
            pool = [spec for spec in compatible if spec not in selected]
        if not pool:
            break
        def rank_key(spec):
            kind = spec["final_answer_kind"]
            exact_kind = kind == target_kind
            compatible_rich = target_kind in rich_kinds and kind in rich_kinds
            return (
                0 if spec["question_type"] == preferred_type and not selected else 1,
                0 if exact_kind else (1 if compatible_rich else 2),
                0 if spec["question_type"] not in used_types else 1,
            )

        pool.sort(key=rank_key)
        best_key = rank_key(pool[0])
        tied = [spec for spec in pool if rank_key(spec) == best_key]
        chosen = rng.choice(tied)
        selected.append(chosen)
        used_types.add(chosen["question_type"])
    return selected[:count]


def validate_template_registry():
    ids = [spec["template_id"] for spec in QUESTION_TEMPLATES]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question template IDs")
    for spec in QUESTION_TEMPLATES:
        if spec["min_regions"] < 2 or spec["max_regions"] < spec["min_regions"]:
            raise ValueError(f"invalid region bounds: {spec['template_id']}")
        if not spec["instruction"].strip():
            raise ValueError(f"empty template instruction: {spec['template_id']}")
        if spec.get("evidence_scope") not in {"any", "single_image", "cross_image"}:
            raise ValueError(f"invalid evidence scope: {spec['template_id']}")
        if spec["final_answer_kind"] not in {
            "anchor", "numeric", "boolean", "semantic", "semantic_set",
            "semantic_pair", "text",
        }:
            raise ValueError(f"invalid answer kind: {spec['template_id']}")
    return True
