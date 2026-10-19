const BASE = "/api";

async function request(path, options = {}) {
  const response = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (response.status === 204) return null;
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      body && typeof body.detail === "string"
        ? body.detail
        : body?.detail
          ? JSON.stringify(body.detail)
          : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return body;
}

export const analyzeNote = (text) =>
  request("/analyze", { method: "POST", body: JSON.stringify({ text }) });

export const listRecords = (patientId) =>
  request(`/records${patientId ? `?patient_id=${patientId}` : ""}`);

export const getRecord = (id) => request(`/records/${id}`);

export const createRecord = (payload) =>
  request("/records", { method: "POST", body: JSON.stringify(payload) });

export const deleteRecord = (id) =>
  request(`/records/${id}`, { method: "DELETE" });

export const getHealth = () => request("/health");
