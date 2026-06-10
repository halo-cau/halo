/**
 * Shared state for the background digital-twin (Pi3 + SAM3) job.
 *
 * The scan page submits a job to POST /api/v1/twin and stores its id here, in localStorage, so the work
 * survives page navigation. The scan page reconnects to it on load, and the cross-page progress badge
 * (lib/twinIndicator) reads the same record to show progress on every other page. Keeping the key, the
 * status shape, and the load/save helpers in one module guarantees the producer and the consumers never
 * drift apart.
 */

/** localStorage key holding the active/last twin job ({jobId, kind}). */
export const TWIN_JOB_KEY = "halo.twin.job";
/** localStorage key holding the jobId whose progress badge the user dismissed (so we stop nagging). */
export const TWIN_DISMISS_KEY = "halo.twin.dismissed";
/** localStorage key holding the registry of COMPLETED scans so multiple rooms survive a scan-page reset. */
export const TWIN_SCANS_KEY = "halo.twin.scans";

export interface SavedJob {
  jobId: string;
  kind: string;
}

/**
 * A completed twin scan. The scan page resets after each reconstruction so a new room can be scanned, but
 * the finished room must not be lost: every completed job is appended here so the dashboard can list every
 * scanned room and open any of them in the editor (in a new tab) while a fresh scan runs concurrently.
 */
export interface ScanRecord {
  jobId: string;
  kind: string;
  name: string; // human label (defaults to the completion timestamp)
  completedAt: number; // epoch ms
}

/** All completed scans, newest first. */
export function listScans(): ScanRecord[] {
  try {
    const v = localStorage.getItem(TWIN_SCANS_KEY);
    return v ? (JSON.parse(v) as ScanRecord[]) : [];
  } catch {
    return [];
  }
}

/** Append a completed scan (idempotent by jobId; newest first, capped at 20). */
export function addScan(rec: ScanRecord): void {
  const list = listScans().filter((s) => s.jobId !== rec.jobId);
  list.unshift(rec);
  localStorage.setItem(TWIN_SCANS_KEY, JSON.stringify(list.slice(0, 20)));
}

/** The most recently completed scan, or null. */
export function latestScan(): ScanRecord | null {
  return listScans()[0] ?? null;
}

/** Mirror of the backend status.json payload returned by GET /api/v1/twin/{id}. */
export interface TwinStatus {
  state: string; // queued | running | done | error
  step: string;
  message?: string;
  pct?: number;
  outputs?: Record<string, string>;
  error?: string;
  precomputed?: boolean; // served from a precomputed sample (demo mode), not a live pipeline run
}

export function loadSavedJob(): SavedJob | null {
  try {
    const v = localStorage.getItem(TWIN_JOB_KEY);
    return v ? (JSON.parse(v) as SavedJob) : null;
  } catch {
    return null;
  }
}

export function saveJob(jobId: string, kind: string): void {
  localStorage.setItem(TWIN_JOB_KEY, JSON.stringify({ jobId, kind }));
}

export function clearJob(): void {
  localStorage.removeItem(TWIN_JOB_KEY);
}
