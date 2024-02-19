// Server-side base URL for the ragGround API. Overridable per environment.
export const API_URL = process.env.RAGGROUND_API_URL ?? "http://localhost:8000";

export type CheckResult = { ok: boolean; latency_ms: number; error: string | null };
export type Readiness = { status: "ready" | "degraded"; checks: Record<string, CheckResult> };

export async function getReadiness(): Promise<Readiness | { status: "unreachable"; error: string }> {
  try {
    const res = await fetch(`${API_URL}/health/ready`, {
      cache: "no-store",
      signal: AbortSignal.timeout(3000),
    });
    return (await res.json()) as Readiness;
  } catch (err) {
    return { status: "unreachable", error: err instanceof Error ? err.message : String(err) };
  }
}
