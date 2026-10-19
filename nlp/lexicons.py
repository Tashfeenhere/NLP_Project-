"""Closed vocabulary for the physiotherapy domain.

Every list here is shared by three consumers:

* the spaCy rule layers (`nlp/rules/*`) build PhraseMatchers from it,
* the synthetic generator (`data/generate_notes.py`) draws slots from it,
* the tests assert coverage against it.

Multi-word entries must come first in any PhraseMatcher construction so that
"lower back" beats "back" and "cervical spine" beats "spine".
"""

# --------------------------------------------------------------------------
# Body parts
# --------------------------------------------------------------------------
# Grouped by joint family -- the generator uses the grouping to cover each
# family systematically.

BODY_PARTS_BY_REGION = {
    "knee": ["knee", "patella", "kneecap", "patellofemoral joint", "knee joint"],
    "lower_back": ["lower back", "low back", "lumbar spine", "lumbar region", "lumbar"],
    "shoulder": [
        "shoulder",
        "shoulder joint",
        "rotator cuff",
        "acromioclavicular joint",
        "glenohumeral joint",
    ],
    "neck": ["neck", "cervical spine", "cervical region", "cervical"],
    "hip": ["hip", "hip joint", "greater trochanter"],
    "ankle": ["ankle", "ankle joint", "achilles", "achilles tendon", "calcaneus"],
}

BODY_PARTS = [p for parts in BODY_PARTS_BY_REGION.values() for p in parts]

# Canonical form for every surface form.  Keys are lower-cased surface strings.
BODY_PART_CANON = {
    "knee": "knee",
    "patella": "knee",
    "kneecap": "knee",
    "knee joint": "knee",
    "patellofemoral joint": "knee",
    "lower back": "lower back",
    "low back": "lower back",
    "lumbar spine": "lower back",
    "lumbar region": "lower back",
    "lumbar": "lower back",
    "shoulder": "shoulder",
    "shoulder joint": "shoulder",
    "rotator cuff": "shoulder",
    "acromioclavicular joint": "shoulder",
    "glenohumeral joint": "shoulder",
    "neck": "neck",
    "cervical spine": "neck",
    "cervical region": "neck",
    "cervical": "neck",
    "hip": "hip",
    "hip joint": "hip",
    "greater trochanter": "hip",
    "ankle": "ankle",
    "ankle joint": "ankle",
    "achilles": "ankle",
    "achilles tendon": "ankle",
    "calcaneus": "ankle",
}

LATERALITY_WORDS = {
    "right": "right",
    "rt": "right",
    "r": "right",
    "left": "left",
    "lt": "left",
    "l": "left",
    "bilateral": "bilateral",
    "bilaterally": "bilateral",
    "both": "bilateral",
    "both sides": "bilateral",
}

# --------------------------------------------------------------------------
# Muscles / soft tissue
# --------------------------------------------------------------------------
MUSCLES = [
    "quadriceps",
    "quadriceps femoris",
    "quads",
    "vastus medialis",
    "vastus lateralis",
    "hamstrings",
    "gluteals",
    "gluteus maximus",
    "gluteus medius",
    "glutes",
    "calf",
    "gastrocnemius",
    "soleus",
    "iliopsoas",
    "hip flexors",
    "adductors",
    "abductors",
    "calf complex",
    "erector spinae",
    "multifidus",
    "latissimus dorsi",
    "deltoid",
    "trapezius",
    "upper trapezius",
    "lower trapezius",
    "rotator cuff muscles",
    "infraspinatus",
    "supraspinatus",
    "subscapularis",
    "pectoralis major",
    "serratus anterior",
    "deep neck flexors",
    "thoracic extensors",
]

MUSCLE_CANON = {
    "quadriceps": "quadriceps",
    "quadriceps femoris": "quadriceps",
    "quads": "quadriceps",
    "vastus medialis": "vastus medialis",
    "vastus lateralis": "vastus lateralis",
    "hamstrings": "hamstrings",
    "gluteals": "gluteals",
    "glutes": "gluteals",
    "gluteus maximus": "gluteus maximus",
    "gluteus medius": "gluteus medius",
    "calf": "calf",
    "calf complex": "calf",
    "gastrocnemius": "gastrocnemius",
    "soleus": "soleus",
    "iliopsoas": "iliopsoas",
    "hip flexors": "hip flexors",
    "adductors": "adductors",
    "abductors": "abductors",
    "erector spinae": "erector spinae",
    "multifidus": "multifidus",
    "latissimus dorsi": "latissimus dorsi",
    "deltoid": "deltoid",
    "trapezius": "trapezius",
    "upper trapezius": "upper trapezius",
    "lower trapezius": "lower trapezius",
    "rotator cuff muscles": "rotator cuff",
    "infraspinatus": "infraspinatus",
    "supraspinatus": "supraspinatus",
    "subscapularis": "subscapularis",
    "pectoralis major": "pectoralis major",
    "serratus anterior": "serratus anterior",
    "deep neck flexors": "deep neck flexors",
    "thoracic extensors": "thoracic extensors",
}

# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------
FINDINGS = [
    "weakness",
    "weak",
    "tightness",
    "tight",
    "effusion",
    "swelling",
    "swollen",
    "stiffness",
    "stiff",
    "tenderness",
    "tender",
    "pain",
    "soreness",
    "sore",
    "instability",
    "unstable",
    "clicking",
    "crepitus",
    "locking",
    "giving way",
    "loss of range",
    "restricted range",
    "decreased range of motion",
    "limited range of motion",
    "muscle guarding",
    "spasm",
    "trigger point",
    "atrophy",
    "wasting",
    "oedema",
    "edema",
    "redness",
    "warmth",
    "paraesthesia",
    "numbness",
    "tingling",
    "antalgic gait",
    "limp",
    "postural deviation",
    "poor control",
    "altered movement pattern",
]

# Bare pain words are treated as the reason for the visit rather than as a
# finding.  Tagging "pain" in "knee pain" as FINDING duplicates the complaint
# and produces false positives on every note, so the rule layer uses this
# narrower list while the canonical map keeps the full vocabulary for NER /
# manual corrections.
RULE_FINDINGS = [f for f in FINDINGS if f not in {"pain", "soreness", "sore"}]

FINDING_CANON = {
    "weakness": "weakness",
    "weak": "weakness",
    "tightness": "tightness",
    "tight": "tightness",
    "effusion": "effusion",
    "swelling": "swelling",
    "swollen": "swelling",
    "stiffness": "stiffness",
    "stiff": "stiffness",
    "tenderness": "tenderness",
    "tender": "tenderness",
    "pain": "pain",
    "soreness": "pain",
    "sore": "pain",
    "instability": "instability",
    "unstable": "instability",
    "clicking": "clicking",
    "crepitus": "crepitus",
    "locking": "locking",
    "giving way": "giving way",
    "loss of range": "loss of range",
    "restricted range": "restricted range",
    "decreased range of motion": "loss of range",
    "limited range of motion": "loss of range",
    "muscle guarding": "muscle guarding",
    "spasm": "spasm",
    "trigger point": "trigger point",
    "atrophy": "atrophy",
    "wasting": "atrophy",
    "oedema": "oedema",
    "edema": "oedema",
    "redness": "redness",
    "warmth": "warmth",
    "paraesthesia": "paraesthesia",
    "numbness": "numbness",
    "tingling": "paraesthesia",
    "antalgic gait": "antalgic gait",
    "limp": "antalgic gait",
    "postural deviation": "postural deviation",
    "poor control": "poor control",
    "altered movement pattern": "altered movement pattern",
}

# --------------------------------------------------------------------------
# Movements (used for ROM binding)
# --------------------------------------------------------------------------
MOVEMENTS = [
    "flexion",
    "extension",
    "abduction",
    "adduction",
    "internal rotation",
    "external rotation",
    "medial rotation",
    "lateral rotation",
    "rotation",
    "inversion",
    "eversion",
    "dorsiflexion",
    "plantarflexion",
    "plantar flexion",
    "pronation",
    "supination",
    "circumduction",
    "side flexion",
    "lateral flexion",
    "retraction",
    "protraction",
    "elevation",
    "extension lag",
]

MOVEMENT_CANON = {
    "flexion": "flexion",
    "extension": "extension",
    "abduction": "abduction",
    "adduction": "adduction",
    "internal rotation": "internal rotation",
    "internal rotation": "internal rotation",
    "medial rotation": "internal rotation",
    "external rotation": "external rotation",
    "lateral rotation": "external rotation",
    "rotation": "rotation",
    "inversion": "inversion",
    "eversion": "eversion",
    "dorsiflexion": "dorsiflexion",
    "plantarflexion": "plantar flexion",
    "plantar flexion": "plantar flexion",
    "pronation": "pronation",
    "supination": "supination",
    "circumduction": "circumduction",
    "side flexion": "side flexion",
    "lateral flexion": "side flexion",
    "retraction": "retraction",
    "protraction": "protraction",
    "elevation": "elevation",
    "extension lag": "extension lag",
}

# Which movements make sense at which canonical body part.  Used when the note
# states a movement without naming a joint ("flexion limited to 90 degrees").
MOVEMENT_JOINT_HINT = {
    "flexion": None,
    "extension": None,
    "abduction": None,
    "adduction": None,
    "internal rotation": None,
    "external rotation": None,
    "rotation": None,
    "inversion": "ankle",
    "eversion": "ankle",
    "dorsiflexion": "ankle",
    "plantar flexion": "ankle",
    "pronation": "ankle",
    "supination": "ankle",
    "circumduction": None,
    "side flexion": "lower back",
    "lateral flexion": "lower back",
    "retraction": "neck",
    "protraction": "neck",
    "elevation": "shoulder",
    "extension lag": "knee",
}

# --------------------------------------------------------------------------
# Exercises
# --------------------------------------------------------------------------
# Matching runs on lemmas, so only the singular lemma is needed here; the
# surface forms are kept for the generator and the canonical-name mapping.
EXERCISES = [
    "straight leg raise",
    "slr",
    "wall squat",
    "wall slide",
    "wall slide exercise",
    "squat",
    "lunge",
    "step up",
    "step down",
    "bridge",
    "glute bridge",
    "single leg bridge",
    "clamshell",
    "leg press",
    "hamstring curl",
    "quadriceps set",
    "quad set",
    "heel slide",
    "terminal knee extension",
    "calf raise",
    "heel raise",
    "single leg balance",
    "balance exercise",
    "step-down exercise",
    "side lying abduction",
    "hip abduction",
    "hip hinge",
    "deadlift",
    "romanian deadlift",
    "bird dog",
    "bird-dog exercise",
    "dead bug",
    "plank",
    "side plank",
    "pallof press",
    "cat camel",
    "cat cow",
    "pelvic tilt",
    "prone press up",
    "prone extension",
    "McKenzie extension",
    "cervical retraction",
    "chin tuck",
    "shoulder shrug",
    "scapular retraction",
    "scapular setting",
    "band pull apart",
    "external rotation exercise",
    "pectoral stretch",
    "towel stretch",
    "doorway stretch",
    "hamstring stretch",
    "calf stretch",
    "quadriceps stretch",
    "hip flexor stretch",
    "piriformis stretch",
    "balance board",
    "wobble board",
    "stationary bike",
    "walking programme",
    "aqua therapy",
    "isometric exercise",
    "eccentric exercise",
]

EXERCISE_CANON = {
    "straight leg raise": "straight leg raise",
    "slr": "straight leg raise",
    "wall squat": "wall squat",
    "wall slide": "wall slide",
    "wall slide exercise": "wall slide",
    "squat": "squat",
    "lunge": "lunge",
    "step up": "step up",
    "step down": "step down",
    "step-down exercise": "step down",
    "bridge": "glute bridge",
    "glute bridge": "glute bridge",
    "single leg bridge": "single leg bridge",
    "clamshell": "clamshell",
    "leg press": "leg press",
    "hamstring curl": "hamstring curl",
    "quadriceps set": "quadriceps set",
    "quad set": "quadriceps set",
    "heel slide": "heel slide",
    "terminal knee extension": "terminal knee extension",
    "calf raise": "calf raise",
    "heel raise": "calf raise",
    "single leg balance": "single leg balance",
    "balance exercise": "single leg balance",
    "side lying abduction": "side lying abduction",
    "hip abduction": "hip abduction",
    "hip hinge": "hip hinge",
    "deadlift": "deadlift",
    "romanian deadlift": "romanian deadlift",
    "bird dog": "bird dog",
    "bird-dog exercise": "bird dog",
    "dead bug": "dead bug",
    "plank": "plank",
    "side plank": "side plank",
    "pallof press": "pallof press",
    "cat camel": "cat camel",
    "cat cow": "cat camel",
    "pelvic tilt": "pelvic tilt",
    "prone press up": "prone press up",
    "prone extension": "prone extension",
    "mckenzie extension": "McKenzie extension",
    "cervical retraction": "cervical retraction",
    "chin tuck": "chin tuck",
    "shoulder shrug": "shoulder shrug",
    "scapular retraction": "scapular retraction",
    "scapular setting": "scapular setting",
    "band pull apart": "band pull apart",
    "external rotation exercise": "external rotation exercise",
    "pectoral stretch": "pectoral stretch",
    "towel stretch": "towel stretch",
    "doorway stretch": "doorway stretch",
    "hamstring stretch": "hamstring stretch",
    "calf stretch": "calf stretch",
    "quadriceps stretch": "quadriceps stretch",
    "hip flexor stretch": "hip flexor stretch",
    "piriformis stretch": "piriformis stretch",
    "balance board": "balance board",
    "wobble board": "balance board",
    "stationary bike": "stationary bike",
    "walking programme": "walking programme",
    "aqua therapy": "aqua therapy",
    "isometric exercise": "isometric exercise",
    "eccentric exercise": "eccentric exercise",
}

# --------------------------------------------------------------------------
# Aggravating / relieving vocabulary
# --------------------------------------------------------------------------
TRIGGERS = [
    "climbing stairs",
    "going up stairs",
    "going down stairs",
    "stair climbing",
    "prolonged sitting",
    "sitting",
    "standing",
    "prolonged standing",
    "walking",
    "running",
    "jogging",
    "squatting",
    "kneeling",
    "bending",
    "lifting",
    "heavy lifting",
    "twisting",
    "bending forward",
    "coughing",
    "sneezing",
    "driving",
    "sleeping",
    "lying on side",
    "lying prone",
    "overhead reaching",
    "reaching overhead",
    "throwing",
    "computer work",
    "desk work",
    "looking down",
    "turning head",
    "hopping",
    "jumping",
    "running downhill",
    "walking downhill",
    "walking on uneven ground",
    "sleeping on side",
    "morning stiffness",
]

TRIGGER_CANON = {t: t for t in TRIGGERS}
TRIGGER_CANON.update(
    {
        "going up stairs": "climbing stairs",
        "going down stairs": "climbing stairs",
        "stair climbing": "climbing stairs",
        "sitting": "prolonged sitting",
        "standing": "prolonged standing",
        "jogging": "running",
        "heavy lifting": "lifting",
        "reaching overhead": "overhead reaching",
        "desk work": "computer work",
        "sleeping on side": "lying on side",
    }
)

RELIEVING_FACTORS = [
    "rest",
    "resting",
    "walking",
    "stretching",
    "heat",
    "ice",
    "icing",
    "heat pack",
    "ice pack",
    "elevation",
    "massage",
    "taping",
    "medication",
    "analgesia",
    "anti-inflammatories",
    "positioning",
    "change of position",
    "gentle movement",
    "lying down",
    "sitting down",
    "foam rolling",
]

RELIEVING_CANON = {
    "rest": "rest",
    "resting": "rest",
    "walking": "walking",
    "stretching": "stretching",
    "heat": "heat",
    "heat pack": "heat",
    "ice": "ice",
    "icing": "ice",
    "ice pack": "ice",
    "elevation": "elevation",
    "massage": "massage",
    "taping": "taping",
    "medication": "medication",
    "analgesia": "medication",
    "anti-inflammatories": "medication",
    "positioning": "positioning",
    "change of position": "change of position",
    "gentle movement": "gentle movement",
    "lying down": "lying down",
    "sitting down": "sitting down",
    "foam rolling": "foam rolling",
}

# --------------------------------------------------------------------------
# Duration / frequency
# --------------------------------------------------------------------------
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "a": 1,
    "an": 1,
    "couple": 2,
    "few": 3,
}

TIME_UNITS = {
    "day": "day",
    "days": "day",
    "week": "week",
    "weeks": "week",
    "month": "month",
    "months": "month",
    "year": "year",
    "years": "year",
}

FREQUENCY_TERMS = [
    "constant",
    "constantly",
    "intermittent",
    "intermittently",
    "occasional",
    "occasionally",
    "frequent",
    "frequently",
    "daily",
    "every day",
    "twice a day",
    "once a day",
    "three times a day",
    "recurrent",
    "episodic",
]

FREQUENCY_CANON = {
    "constant": "constant",
    "constantly": "constant",
    "intermittent": "intermittent",
    "intermittently": "intermittent",
    "occasional": "intermittent",
    "occasionally": "intermittent",
    "frequent": "frequent",
    "frequently": "frequent",
    "daily": "daily",
    "every day": "daily",
    "twice a day": "twice daily",
    "once a day": "once daily",
    "three times a day": "three times daily",
    "recurrent": "recurrent",
    "episodic": "episodic",
}

# --------------------------------------------------------------------------
# Qualitative ROM descriptions
# --------------------------------------------------------------------------
ROM_QUALITY_WORDS = {
    "reduced": "reduced",
    "decreased": "reduced",
    "limited": "limited",
    "restricted": "limited",
    "full": "full",
    "normal": "normal",
    "improved": "improved",
    "painful": "painful",
    "blocked": "blocked",
}

# --------------------------------------------------------------------------
# Negation cues
# --------------------------------------------------------------------------
NEGATION_CUES = [
    "no",
    "not",
    "without",
    "denies",
    "denied",
    "negative for",
    "negative",
    "absence of",
    "absent",
    "none",
    "nil",
    "free of",
    "unremarkable",
]

# Words that flip a negation back ("not without pain" is awkward but seen).
NEGATION_SCOPE_BREAKERS = ["but", "however", "although", "though", "except"]

# --------------------------------------------------------------------------
# Convenience: every gazetteer built from this module
# --------------------------------------------------------------------------
GAZETTEERS = {
    "BODY_PART": BODY_PARTS,
    "MUSCLE": MUSCLES,
    "FINDING": FINDINGS,
    "EXERCISE": EXERCISES,
    "MOVEMENT": MOVEMENTS,
    "RELIEVING_FACTOR": RELIEVING_FACTORS,
    "FREQUENCY": FREQUENCY_TERMS,
}

CANONICAL_MAPS = {
    "BODY_PART": BODY_PART_CANON,
    "MUSCLE": MUSCLE_CANON,
    "FINDING": FINDING_CANON,
    "EXERCISE": EXERCISE_CANON,
    "MOVEMENT": MOVEMENT_CANON,
    "RELIEVING_FACTOR": RELIEVING_CANON,
    "FREQUENCY": FREQUENCY_CANON,
}


def canonical(label: str, surface: str) -> str:
    """Return the canonical form for a surface string under ``label``.

    Falls back to a lower-cased, whitespace-collapsed version of the surface
    so unknown NER output still normalizes predictably.
    """
    text = " ".join(surface.lower().split())
    mapping = CANONICAL_MAPS.get(label)
    if mapping and text in mapping:
        return mapping[text]
    # Plural fallback: "knees" -> "knee", "patellofemoral joints" ->
    # "patellofemoral joint".  Only applied when the singular is a known key,
    # so arbitrary words are never mangled.
    if mapping and text.endswith("s") and text[:-1] in mapping:
        return mapping[text[:-1]]
    return text
