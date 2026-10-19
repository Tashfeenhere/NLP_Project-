export default function SavedRecordsPanel({
  records,
  selectedId,
  onOpen,
  onDelete,
  onRefresh,
  busy,
}) {
  return (
    <div className="card">
      <div className="card-head">
        <h2>Confirmed records</h2>
        <button type="button" className="small" onClick={onRefresh} disabled={busy}>
          Refresh
        </button>
      </div>

      {records.length === 0 ? (
        <p className="muted">No records saved yet.</p>
      ) : (
        <ul className="record-list">
          {records.map((record) => (
            <li
              key={record.id}
              className={record.id === selectedId ? "selected" : undefined}
            >
              <div className="record-main">
                <div className="record-line">
                  <strong>#{record.id}</strong>
                  <span>
                    {record.body_part
                      ? record.body_part
                      : "body part not set"}
                    {record.pain_severity != null
                      ? ` · pain ${record.pain_severity}/10`
                      : ""}
                  </span>
                  {record.session_date && (
                    <span className="record-date">{record.session_date}</span>
                  )}
                </div>
                <p className="record-preview">{record.preview}</p>
                {record.patient_name && (
                  <p className="record-patient">{record.patient_name}</p>
                )}
              </div>
              <div className="record-actions">
                <button
                  type="button"
                  className="small"
                  onClick={() => onOpen(record.id)}
                  disabled={busy}
                >
                  Open
                </button>
                <button
                  type="button"
                  className="small danger"
                  onClick={() => {
                    if (window.confirm(`Delete record #${record.id}?`)) {
                      onDelete(record.id);
                    }
                  }}
                  disabled={busy}
                >
                  Delete
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
