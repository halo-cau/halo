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

export interface SavedJob {
  jobId: string;
  kind: string;
}

/** Mirror of the backend status.json payload returned by GET /api/v1/twin/{id}. */
export interface TwinStatus {
  state: string; // queued | running | done | error
  step: string;
  message?: string;
  pct?: number;
  outputs?: Record<string, string>;
  error?: string;
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
