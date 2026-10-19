export default function SaveBar({
  patientName,
  onPatientName,
  sessionDate,
  onSessionDate,
  onSave,
  busy,
}) {
  return (
    <div className="card savebar">
      <div className="card-head">
        <h2>Confirm record</h2>
        <span className="hint">Saving stores the edited fields, not the raw prediction.</span>
      </div>
      <div className="save-grid">
        <label className="field">
          <span>Patient</span>
          <input
            value={patientName}
            placeholder="optional"
            onChange={(event) => onPatientName(event.target.value)}
          />
        </label>
        <label className="field">
          <span>Session date</span>
          <input
            type="date"
            value={sessionDate}
            onChange={(event) => onSessionDate(event.target.value)}
          />
        </label>
        <button type="button" className="primary" onClick={onSave} disabled={busy}>
          {busy ? "Saving…" : "Confirm & save"}
        </button>
      </div>
    </div>
  );
}
