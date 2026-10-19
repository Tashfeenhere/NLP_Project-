import { useCallback, useEffect, useState } from "react";
import NoteForm from "./components/NoteForm.jsx";
import EntityHighlighter from "./components/EntityHighlighter.jsx";
import FieldsTable from "./components/FieldsTable.jsx";
import SaveBar from "./components/SaveBar.jsx";
import SavedRecordsPanel from "./components/SavedRecordsPanel.jsx";
import {
  analyzeNote,
  createRecord,
  deleteRecord,
  getHealth,
  getRecord,
  listRecords,
} from "./api.js";

const today = () => new Date().toISOString().slice(0, 10);

export default function App() {
  const [note, setNote] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [fields, setFields] = useState(null);
  const [patientName, setPatientName] = useState("");
  const [sessionDate, setSessionDate] = useState(today);
  const [selectedId, setSelectedId] = useState(null);
  const [saved, setSaved] = useState([]);
  const [health, setHealth] = useState(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState(null);

  const refreshSaved = useCallback(async () => {
    try {
      setSaved(await listRecords());
    } catch (error) {
      setStatus({ type: "error", message: error.message });
    }
  }, []);

  useEffect(() => {
    refreshSaved();
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, [refreshSaved]);

  async function handleAnalyze() {
    setBusy(true);
    setStatus(null);
    setSelectedId(null);
    try {
      const result = await analyzeNote(note);
      setAnalysis(result);
      setFields(result.fields);
      if (!result.ready) {
        setStatus({ type: "warn", message: result.warnings.join(" ") });
      }
    } catch (error) {
      setStatus({ type: "error", message: error.message });
    } finally {
      setBusy(false);
    }
  }

  async function handleSave() {
    if (!analysis || !fields) return;
    setBusy(true);
    setStatus(null);
    try {
      const result = await createRecord({
        patient_name: patientName.trim() || null,
        session_date: sessionDate || null,
        text: note,
        fields,
        entities: analysis.entities,
        negated_entities: analysis.negated_entities,
        model_version: analysis.model_version,
      });
      setSelectedId(result.id);
      setStatus({ type: "ok", message: `Record #${result.id} saved.` });
      await refreshSaved();
    } catch (error) {
      setStatus({ type: "error", message: error.message });
    } finally {
      setBusy(false);
    }
  }

  async function handleOpen(id) {
    setBusy(true);
    setStatus(null);
    try {
      const record = await getRecord(id);
      setNote(record.text);
      setFields(record.fields);
      setPatientName(record.patient_name || "");
      setSessionDate(record.session_date || today());
      setSelectedId(record.id);
      setAnalysis({
        ready: true,
        warnings: [],
        entities: record.entities,
        negated_entities: record.negated_entities,
        model_version: record.model_version,
      });
    } catch (error) {
      setStatus({ type: "error", message: error.message });
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(id) {
    setBusy(true);
    try {
      await deleteRecord(id);
      if (selectedId === id) {
        setSelectedId(null);
        setAnalysis(null);
        setFields(null);
      }
      setStatus({ type: "ok", message: `Record #${id} deleted.` });
      await refreshSaved();
    } catch (error) {
      setStatus({ type: "error", message: error.message });
    } finally {
      setBusy(false);
    }
  }

  function handleNew() {
    setNote("");
    setAnalysis(null);
    setFields(null);
    setSelectedId(null);
    setPatientName("");
    setSessionDate(today());
    setStatus(null);
  }

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>Clinical Note Analyzer</h1>
          <p className="subtitle">
            Rules + trained NER extraction for physiotherapy notes
          </p>
        </div>
        <div className="model-chip" title="Model loaded by the backend">
          {health ? health.model_version : "backend offline"}
        </div>
      </header>

      <main className="layout">
        <section className="column">
          <NoteForm
            note={note}
            onChange={setNote}
            onAnalyze={handleAnalyze}
            onNew={handleNew}
            busy={busy}
          />

          {analysis && (
            <div className="card">
              <div className="card-head">
                <h2>Source note</h2>
                <span className="hint">
                  Highlighted spans come from the extraction; hover for provenance.
                </span>
              </div>
              <EntityHighlighter
                text={note}
                entities={analysis.entities}
                negated={analysis.negated_entities}
              />
            </div>
          )}

          {analysis && analysis.negated_entities.length > 0 && (
            <div className="card">
              <div className="card-head">
                <h2>Negated mentions</h2>
                <span className="hint">Excluded from the confirmed fields.</span>
              </div>
              <ul className="negated-list">
                {analysis.negated_entities.map((entity, index) => (
                  <li key={`${entity.start}-${index}`}>
                    <strong>{entity.label}</strong> {entity.text}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {analysis?.warnings?.length > 0 && (
            <div className="warnings">
              {analysis.warnings.map((warning) => (
                <p key={warning}>{warning}</p>
              ))}
            </div>
          )}
        </section>

        <section className="column">
          {fields ? (
            <>
              <div className="card">
                <div className="card-head">
                  <h2>Extracted fields</h2>
                  <span className="hint">Edit anything before confirming.</span>
                </div>
                <FieldsTable fields={fields} onChange={setFields} />
              </div>
              <SaveBar
                patientName={patientName}
                onPatientName={setPatientName}
                sessionDate={sessionDate}
                onSessionDate={setSessionDate}
                onSave={handleSave}
                busy={busy}
              />
            </>
          ) : (
            <div className="card empty-state">
              <h2>No analysis yet</h2>
              <p>
                Paste a note on the left and press <strong>Analyze note</strong>.
                The extracted fields will appear here for review.
              </p>
            </div>
          )}

          <SavedRecordsPanel
            records={saved}
            selectedId={selectedId}
            onOpen={handleOpen}
            onDelete={handleDelete}
            onRefresh={refreshSaved}
            busy={busy}
          />
        </section>
      </main>

      {status && <div className={`status ${status.type}`}>{status.message}</div>}

      <footer className="disclaimer">
        This tool assists documentation only — it does not diagnose. All
        extracted values must be checked by the treating clinician.
      </footer>
    </div>
  );
}
