// Backend address. `||` (not `??`) so an empty VITE_API_URL= also falls back to the default.
export const API_URL = (
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");
