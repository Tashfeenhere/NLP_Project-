import { useState } from "react";

const SCALARS = [
  ["body_part", "Body part"],
  ["duration", "Duration"],
  ["frequency", "Frequency"],
];

const LIST_FIELDS = [
  ["triggers", "Triggers"],
  ["relieving_factors", "Relieving factors"],
  ["muscles", "Muscles"],
  ["findings", "Findings"],
  ["exercises", "Exercises"],
];

const TEXT_FIELDS = [...LIST_FIELDS.map(([key, label]) => [key, label])];

export default function FieldsTable({ fields, onChange }) {
  const [addLabel, setAddLabel] = useState("triggers");
  const [addValue, setAddValue] = useState("");

  const update = (patch) => onChange({ ...fields, ...patch });
  const pain = fields.pain_severity || { raw: "", value: null };

  const setPain = (patch) =>
    update({ pain_severity: { ...pain, ...patch } });

  const setRom = (index, patch) =>
    update({
      rom: (fields.rom || []).map((row, rowIndex) =>
        rowIndex === index ? { ...row, ...patch } : row,
      ),
    });

  const removeRom = (index) =>
    update({ rom: (fields.rom || []).filter((_, rowIndex) => rowIndex !== index) });

  const setListItem = (key, index, value) =>
    update({
      [key]: (fields[key] || []).map((item, itemIndex) =>
        itemIndex === index ? value : item,
      ),
    });

  const removeListItem = (key, index) =>
    update({ [key]: (fields[key] || []).filter((_, itemIndex) => itemIndex !== index) });

  function addField() {
    const value = addValue.trim();
    if (addLabel === "rom") {
      update({
        rom: [
          ...(fields.rom || []),
          { movement: value || "flexion", value: null, unit: "deg", quality: null },
        ],
      });
    } else {
      if (!value) return;
      update({ [addLabel]: [...(fields[addLabel] || []), value] });
    }
    setAddValue("");
  }

  return (
    <div className="fields">
      <div className="field-grid">
        {SCALARS.slice(0, 1).map(([key, label]) => (
          <label key={key} className="field">
            <span>{label}</span>
            <input
              value={fields[key] || ""}
              placeholder="not found"
              onChange={(event) => update({ [key]: event.target.value || null })}
            />
          </label>
        ))}

        <label className="field">
          <span>Laterality</span>
          <select
            value={fields.laterality || ""}
            onChange={(event) => update({ laterality: event.target.value || null })}
          >
            <option value="">not found</option>
            <option value="right">right</option>
            <option value="left">left</option>
            <option value="bilateral">bilateral</option>
          </select>
        </label>

        <label className="field">
          <span>Pain (raw)</span>
          <input
            value={pain.raw || ""}
            placeholder="e.g. 6/10"
            onChange={(event) =>
              setPain({ raw: event.target.value, range: null })
            }
          />
        </label>

        <label className="field">
          <span>Pain (0–10)</span>
          <input
            type="number"
            min="0"
            max="10"
            value={pain.value ?? ""}
            placeholder="—"
            onChange={(event) =>
              setPain({
                value: event.target.value === "" ? null : Number(event.target.value),
              })
            }
          />
        </label>

        {SCALARS.slice(1).map(([key, label]) => (
          <label key={key} className="field">
            <span>{label}</span>
            <input
              value={fields[key] || ""}
              placeholder="not found"
              onChange={(event) => update({ [key]: event.target.value || null })}
            />
          </label>
        ))}
      </div>

      <div className="subsection">
        <div className="subsection-head">
          <h3>Range of motion</h3>
          <button
            type="button"
            className="small"
            onClick={() =>
              update({
                rom: [
                  ...(fields.rom || []),
                  { movement: "flexion", value: null, unit: "deg", quality: null },
                ],
              })
            }
          >
            + ROM row
          </button>
        </div>
        {(fields.rom || []).length === 0 && <p className="muted">No ROM measured.</p>}
        {(fields.rom || []).map((row, index) => (
          <div className="rom-row" key={index}>
            <input
              aria-label="movement"
              value={row.movement || ""}
              onChange={(event) => setRom(index, { movement: event.target.value })}
            />
            <input
              aria-label="value"
              type="number"
              value={row.value ?? ""}
              placeholder="—"
              onChange={(event) =>
                setRom(index, {
                  value: event.target.value === "" ? null : Number(event.target.value),
                })
              }
            />
            <input
              aria-label="unit"
              value={row.unit || ""}
              placeholder="deg"
              onChange={(event) => setRom(index, { unit: event.target.value || null })}
            />
            <input
              aria-label="quality"
              value={row.quality || ""}
              placeholder="quality (reduced, full…)"
              onChange={(event) =>
                setRom(index, { quality: event.target.value || null })
              }
            />
            <button
              type="button"
              className="small danger"
              onClick={() => removeRom(index)}
              aria-label="delete ROM row"
            >
              ×
            </button>
          </div>
        ))}
      </div>

      {LIST_FIELDS.map(([key, label]) => (
        <div className="subsection" key={key}>
          <div className="subsection-head">
            <h3>{label}</h3>
            <span className="count">{(fields[key] || []).length}</span>
          </div>
          {(fields[key] || []).length === 0 && <p className="muted">None.</p>}
          {(fields[key] || []).map((item, index) => (
            <div className="list-row" key={`${key}-${index}`}>
              <input
                value={item}
                onChange={(event) => setListItem(key, index, event.target.value)}
              />
              <button
                type="button"
                className="small danger"
                onClick={() => removeListItem(key, index)}
                aria-label={`delete ${label} item`}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      ))}

      <div className="add-field">
        <select value={addLabel} onChange={(event) => setAddLabel(event.target.value)}>
          {TEXT_FIELDS.map(([key, label]) => (
            <option key={key} value={key}>
              {label}
            </option>
          ))}
          <option value="rom">ROM movement</option>
        </select>
        <input
          value={addValue}
          placeholder={addLabel === "rom" ? "movement, e.g. abduction" : "value"}
          onChange={(event) => setAddValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") addField();
          }}
        />
        <button type="button" className="small" onClick={addField}>
          Add field
        </button>
      </div>
    </div>
  );
}
