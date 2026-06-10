/**
 * Cross-page progress badge for the background digital-twin (Pi3 + SAM3) job.
 *
 * The scan page submits the reconstruction and records the job in localStorage (lib/twinJob); the GPU
 * work runs server-side in an isolated subprocess. This module mounts a small floating badge on every
 * OTHER page so the user can leave the scan page and still watch progress, get a "twin ready" link back
 * to the viewer, or see an error. It is self-contained — no three.js, no shared CSS — so any page's entry
 * script can pull it in with a side-effect import: `import "./lib/twinIndicator";`.
 */
import { clearJob, loadSavedJob, TWIN_DISMISS_KEY, type TwinStatus } from "./twinJob";

const POLL_MS = 2500;

function injectStyleOnce(): void {
  if (document.getElementById("twin-indicator-style")) return;
  const style = document.createElement("style");
  style.id = "twin-indicator-style";
  style.textContent = `
    #twin-indicator{position:fixed;right:16px;bottom:16px;z-index:9999;width:260px;
      font:13px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;color:#2b2a26;background:#fff;
      border:1px solid #e2dfd6;border-radius:10px;box-shadow:0 6px 24px rgba(0,0,0,.12);padding:10px 12px;
      display:none}
    #twin-indicator.show{display:block}
    #twin-indicator .ti-row{display:flex;align-items:center;gap:8px}
    #twin-indicator .ti-dot{width:9px;height:9px;border-radius:50%;flex:0 0 auto;background:#378add}
    #twin-indicator.done .ti-dot{background:#1d9e75}
    #twin-indicator.error .ti-dot{background:#e24b4a}
    #twin-indicator .ti-dot.spin{animation:ti-pulse 1s ease-in-out infinite}
    @keyframes ti-pulse{0%,100%{opacity:.3}50%{opacity:1}}
    #twin-indicator .ti-title{font-weight:600;flex:1 1 auto;white-space:nowrap;overflow:hidden;
      text-overflow:ellipsis}
    #twin-indicator .ti-msg{color:#6b685f;margin-top:2px;white-space:nowrap;overflow:hidden;
      text-overflow:ellipsis}
    #twin-indicator .ti-bar{height:4px;border-radius:3px;background:#eee9df;margin-top:8px;overflow:hidden}
    #twin-indicator .ti-bar>i{display:block;height:100%;width:0;background:#378add;transition:width .3s}
    #twin-indicator.done .ti-bar>i{background:#1d9e75}
    #twin-indicator a.ti-view{color:#378add;font-weight:600;text-decoration:none}
    #twin-indicator button.ti-x{border:0;background:none;color:#9a978e;cursor:pointer;font-size:16px;
      line-height:1;padding:0 2px}
  `;
  document.head.appendChild(style);
}

export function mountTwinIndicator(): void {
  const job = loadSavedJob();
  if (!job) return; // nothing submitted yet on this browser
  if (localStorage.getItem(TWIN_DISMISS_KEY) === job.jobId) return; // user dismissed this job's badge
  if (document.getElementById("twin-indicator")) return; // already mounted on this page

  injectStyleOnce();
  const el = document.createElement("div");
  el.id = "twin-indicator";
  el.className = "show";
  el.innerHTML = `
    <div class="ti-row">
      <span class="ti-dot spin"></span>
      <span class="ti-title">Building digital twin…</span>
      <a class="ti-view" href="./scan.html" style="display:none">View</a>
      <button class="ti-x" title="dismiss" type="button">×</button>
    </div>
    <div class="ti-msg"></div>
    <div class="ti-bar"><i></i></div>
  `;
  document.body.appendChild(el);

  const dot = el.querySelector(".ti-dot") as HTMLElement;
  const title = el.querySelector(".ti-title") as HTMLElement;
  const msg = el.querySelector(".ti-msg") as HTMLElement;
  const fill = el.querySelector(".ti-bar > i") as HTMLElement;
  const view = el.querySelector(".ti-view") as HTMLAnchorElement;
  const xBtn = el.querySelector(".ti-x") as HTMLButtonElement;

  let timer: number | undefined;
  xBtn.addEventListener("click", () => {
    localStorage.setItem(TWIN_DISMISS_KEY, job.jobId); // keep the restore pointer; just stop the badge
    if (timer) window.clearTimeout(timer);
    el.remove();
  });

  async function tick(): Promise<void> {
    let s: TwinStatus;
    try {
      const resp = await fetch(`/api/v1/twin/${job!.jobId}`);
      if (resp.status === 404) {
        clearJob();
        el.remove();
        return;
      }
      s = (await resp.json()) as TwinStatus;
    } catch {
      timer = window.setTimeout(tick, POLL_MS); // transient network hiccup — keep trying
      return;
    }

    if (s.state === "done") {
      el.classList.remove("error");
      el.classList.add("done");
      dot.classList.remove("spin");
      title.textContent = "Digital twin ready";
      msg.textContent = "Reconstruction finished.";
      fill.style.width = "100%";
      view.style.display = "inline";
      return; // terminal — stop polling
    }
    if (s.state === "error") {
      el.classList.remove("done");
      el.classList.add("error");
      dot.classList.remove("spin");
      title.textContent = "Reconstruction failed";
      msg.textContent = s.error || s.message || "pipeline error";
      fill.style.width = "100%";
      return; // terminal — stop polling
    }

    // queued / running
    title.textContent = "Building digital twin…";
    msg.textContent = `${s.step ?? ""}${s.message ? ` · ${s.message}` : ""}`.trim() || "working…";
    fill.style.width = `${Math.max(0, Math.min(100, s.pct ?? 0))}%`;
    timer = window.setTimeout(tick, POLL_MS);
  }

  void tick();
}

// Auto-mount on import so a page only needs `import "./lib/twinIndicator";`.
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => mountTwinIndicator());
} else {
  mountTwinIndicator();
}
