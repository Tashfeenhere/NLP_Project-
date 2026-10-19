const LABELS = {
  BODY_PART: "Body part",
  PAIN_SEVERITY: "Pain",
  ROM: "ROM",
  MUSCLE: "Muscle",
  FINDING: "Finding",
  EXERCISE: "Exercise",
  TRIGGER: "Trigger",
  RELIEVING_FACTOR: "Relieving factor",
  DURATION: "Duration",
  FREQUENCY: "Frequency",
};

/**
 * Render the note with non-overlapping spans.
 *
 * The backend already returns a non-overlapping entity list; this guards
 * again and sorts longest-first so a future change cannot corrupt the offset
 * mapping (dropping overlapped spans instead of slicing mid-tag).
 */
export default function EntityHighlighter({ text, entities = [], negated = [] }) {
  const spans = [...entities, ...negated.map((e) => ({ ...e, negated: true }))]
    .filter((e) => e.start >= 0 && e.end > e.start && e.end <= text.length)
    .sort((a, b) => a.start - b.start || b.end - a.end);

  const nodes = [];
  let cursor = 0;
  for (const span of spans) {
    if (span.start < cursor) continue;
    if (span.start > cursor) nodes.push(text.slice(cursor, span.start));
    nodes.push(
      <mark
        key={`${span.start}-${span.end}-${span.label}`}
        className={span.negated ? "negated" : "entity"}
        data-label={span.label}
        title={`${LABELS[span.label] || span.label} · ${span.source}${
          span.normalized && span.normalized !== span.text.toLowerCase()
            ? ` · ${span.normalized}`
            : ""
        }${span.negated ? " · negated" : ""}`}
      >
        {text.slice(span.start, span.end)}
      </mark>,
    );
    cursor = span.end;
  }
  if (cursor < text.length) nodes.push(text.slice(cursor));

  return <div className="highlighted-note">{nodes}</div>;
}
