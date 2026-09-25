import type {
  AnalyzeScanRequest,
  AnalyzeScanResponse,
  CollectScanRequest,
  DiscoveryResponse,
  InsightsResponse,
  IssuesResponse,
  MetricsResponse,
  ReviewsQuery,
  ReviewsPage,
  Scan,
  ScansResponse,
} from "./types";

const API_BASE_URL = (
  import.meta.env.VITE_API_URL || (import.meta.env.DEV ? "http://localhost:8000" : "")
).replace(/\/$/, "");

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(message: string, status: number, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Network request failed";
    throw new ApiError(message, 0, error);
  }

  const contentType = response.headers.get("content-type") || "";
  const body: unknown = response.status === 204
    ? undefined
    : contentType.includes("application/json")
      ? await response.json().catch(() => undefined)
      : await response.text().catch(() => "");

  if (!response.ok) {
    const detail = body && typeof body === "object" && "detail" in body
      ? (body as { detail: unknown }).detail
      : body;
    const message = typeof detail === "string"
      ? detail
      : Array.isArray(detail)
        ? detail.map((item) => typeof item === "object" && item && "msg" in item ? String(item.msg) : String(item)).join("; ")
        : `Request failed (${response.status})`;
    throw new ApiError(message, response.status, detail);
  }

  return body as T;
}

function queryString(values: object): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(values as Record<string, string | number | boolean | undefined>)) {
    if (value !== undefined && value !== "") params.set(key, String(value));
  }
  const query = params.toString();
  return query ? `?${query}` : "";
}

export function createScan(payload: CollectScanRequest): Promise<Scan> {
  return request<Scan>("/api/scans", { method: "POST", body: JSON.stringify(payload) });
}

export function analyzeScan(scanId: string, payload: AnalyzeScanRequest = {}): Promise<AnalyzeScanResponse> {
  return request<AnalyzeScanResponse>(`/api/scans/${encodeURIComponent(scanId)}/analyze`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getScan(scanId: string): Promise<Scan> {
  return request<Scan>(`/api/scans/${encodeURIComponent(scanId)}`);
}

export function getScans(): Promise<ScansResponse> {
  return request<ScansResponse>("/api/scans");
}

export function getMetrics(scanId: string): Promise<MetricsResponse> {
  return request<MetricsResponse>(`/api/scans/${encodeURIComponent(scanId)}/metrics`);
}

export function getInsights(scanId: string): Promise<InsightsResponse> {
  return request<InsightsResponse>(`/api/scans/${encodeURIComponent(scanId)}/insights`);
}

export function getIssues(scanId: string): Promise<IssuesResponse> {
  return request<IssuesResponse>(`/api/scans/${encodeURIComponent(scanId)}/issues`);
}

export function getReviews(scanId: string, filters: ReviewsQuery = {}): Promise<ReviewsPage> {
  return request<ReviewsPage>(
    `/api/scans/${encodeURIComponent(scanId)}/reviews${queryString(filters)}`,
  );
}

export function downloadReviewsUrl(scanId: string): string {
  return `${API_BASE_URL}/api/scans/${encodeURIComponent(scanId)}/reviews/download`;
}

export function downloadReportUrl(scanId: string): string {
  return `${API_BASE_URL}/api/scans/${encodeURIComponent(scanId)}/report/download`;
}

export function downloadPdfReportUrl(scanId: string): string {
  return `${API_BASE_URL}/api/scans/${encodeURIComponent(scanId)}/report/download.pdf`;
}

export function getDiscovery(appId: string, forceRefresh = false): Promise<DiscoveryResponse> {
  return request<DiscoveryResponse>(
    `/api/apps/${encodeURIComponent(appId)}/discovery${queryString({ force_refresh: forceRefresh || undefined })}`,
  );
}

export { API_BASE_URL };
