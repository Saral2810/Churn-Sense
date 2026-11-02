import type {
  Analytics,
  BatchResponse,
  CustomerInput,
  CustomerPage,
  Health,
  ModelCard,
  Prediction,
  RoiResult,
} from "./types";

export const API_URL = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/\/$/, "");
const API_KEY = import.meta.env.VITE_API_KEY as string | undefined;

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function formatDetail(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail))
    return detail
      .map((d: { loc?: unknown[]; msg?: string }) => `${(d.loc ?? []).slice(1).join(".")}: ${d.msg}`)
      .join("; ");
  return "Request failed";
}

async function request<T>(path: string, init: RequestInit = {}, timeoutMs = 60_000): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const headers = new Headers(init.headers);
  if (API_KEY) headers.set("X-API-Key", API_KEY);
  if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  try {
    const res = await fetch(`${API_URL}${path}`, { ...init, headers, signal: init.signal ?? controller.signal });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new ApiError(res.status, formatDetail(body.detail) || res.statusText);
    }
    return (await res.json()) as T;
  } catch (e) {
    if (e instanceof ApiError) throw e;
    if ((e as Error).name === "AbortError") throw new ApiError(0, "The API took too long to respond");
    throw new ApiError(0, "Can't reach the API");
  } finally {
    clearTimeout(timer);
  }
}

const qs = (params: Record<string, string | number | undefined>) => {
  const p = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => v !== undefined && v !== "" && p.set(k, String(v)));
  return p.toString();
};

export const api = {
  health: () => request<Health>("/health", {}, 75_000),
  analytics: () => request<Analytics>("/api/v1/analytics"),
  model: () => request<ModelCard>("/api/v1/model"),
  predict: (c: CustomerInput) => request<Prediction>("/api/v1/predict", { method: "POST", body: JSON.stringify(c) }),
  roi: (p: { retention_cost: number; revenue_saved: number; success_rate: number; threshold: number }) =>
    request<RoiResult>(`/api/v1/roi?${qs(p)}`),
  customers: (p: { tier?: string; segment?: string; q?: string; sort?: string; limit: number; offset: number }) =>
    request<CustomerPage>(`/api/v1/customers?${qs(p)}`),
  predictCsv: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<BatchResponse>("/api/v1/predict/csv", { method: "POST", body: fd }, 120_000);
  },
};
