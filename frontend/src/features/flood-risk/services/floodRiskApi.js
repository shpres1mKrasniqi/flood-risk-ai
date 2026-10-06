import { API_URL } from "../../../../../config";

export class ApiError extends Error {
  constructor(kind, detail, status) {
    super(detail);
    this.kind = kind;
    this.detail = detail;
    this.status = status;
  }
}

function readDetail(body) {
  const detail = body?.detail;
  if (Array.isArray(detail)) {
    return detail
      .map((e) => `${(e.loc ?? []).slice(1).join(".")}: ${e.msg}`)
      .join("; ");
  }
  return typeof detail === "string" ? detail : "";
}

async function request(path, options) {
  let response;
  try {
    response = await fetch(`${API_URL}${path}`, options);
  } catch {
    throw new ApiError("unreachable", API_URL);
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const kind = response.status === 422 ? "validation" : "server";
    throw new ApiError(
      kind,
      readDetail(body) || `HTTP ${response.status}`,
      response.status,
    );
  }

  if (body === null)
    throw new ApiError(
      "server",
      `${API_URL}${path} did not return JSON. Check VITE_API_URL.`,
    );
  return body;
}

export const getMetadata = () => request("/api/flood-risk/metadata");

export const explainFloodRisk = (payload, language) =>
  request(`/api/flood-risk/explain?language=${encodeURIComponent(language)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
