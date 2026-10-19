const SAMPLES = [
  {
    label: "Knee (canonical)",
    text:
      "Right knee pain rated 6 out of 10, constant for three weeks. " +
      "Worse with climbing stairs. Rest helps. " +
      "Knee flexion is limited to 95 degrees. Quadriceps weakness. " +
      "Prescribed straight leg raises and clamshells.",
  },
  {
    label: "Shoulder (reversed ROM)",
    text:
      "Bilateral rotator cuff pain 7/10. Overhead reaching aggravates it. " +
      "ROM: 90 deg external rotation, shoulder. Rotator cuff muscles weakness. " +
      "Advised wall slides and band pull aparts.",
  },
  {
    label: "Lower back (qualitative)",
    text:
      "Lower back pain 4-5/10 for two months, intermittent. " +
      "Aggravated by prolonged sitting, better with walking. " +
      "Lumbar flexion is reduced. Gluteal tightness. " +
      "Prescribed bird dog, dead bug and glute bridge.",
  },
];

export default function NoteForm({ note, onChange, onAnalyze, onNew, busy }) {
  return (
    <div className="card">
      <div className="card-head">
        <h2>Clinical note</h2>
        <span className="hint">{note.length} characters</span>
      </div>

      <textarea
        className="note-input"
        value={note}
        onChange={(event) => onChange(event.target.value)}
        placeholder="Paste a physiotherapy note, e.g. “Right knee pain 6/10, worse with stairs...”"
        spellCheck={false}
      />

      <div className="sample-row">
        <span className="sample-label">Try a sample:</span>
        {SAMPLES.map((sample) => (
          <button
            key={sample.label}
            type="button"
            className="link-button"
            onClick={() => onChange(sample.text)}
          >
            {sample.label}
          </button>
        ))}
      </div>

      <div className="button-row">
        <button
          type="button"
          className="primary"
          onClick={onAnalyze}
          disabled={busy || !note.trim()}
        >
          {busy ? "Working…" : "Analyze note"}
        </button>
        <button type="button" className="ghost" onClick={onNew} disabled={busy}>
          Clear
        </button>
      </div>
    </div>
  );
}
