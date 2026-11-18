// "Done image" helpers: confirm the image on screen and carry its regions to the next frames.
// Pure functions over serialized results so they can be tested without the editor.

export const DEFAULT_SETTINGS = { n: 3, threshold: 0.8, auto: true };
const KEY = "tasdf.propagation";

const clamp = (v, lo, hi, fallback) => (Number.isFinite(v) ? Math.min(Math.max(v, lo), hi) : fallback);

export const loadSettings = (storage = globalThis.localStorage) => {
  try {
    const raw = JSON.parse(storage.getItem(KEY) || "{}");
    return {
      n: Math.round(clamp(Number(raw.n ?? DEFAULT_SETTINGS.n), 0, 10, DEFAULT_SETTINGS.n)),
      threshold: clamp(Number(raw.threshold ?? DEFAULT_SETTINGS.threshold), 0, 1, DEFAULT_SETTINGS.threshold),
      auto: typeof raw.auto === "boolean" ? raw.auto : DEFAULT_SETTINGS.auto,
    };
  } catch (_error) {
    return { ...DEFAULT_SETTINGS };
  }
};

export const saveSettings = (settings, storage = globalThis.localStorage) => {
  try {
    storage.setItem(KEY, JSON.stringify(settings));
  } catch (_error) {
    // private window or blocked storage: settings just do not persist
  }
};

export const onTarget = (r, target) =>
  r.to_name === target.to_name && (target.item_index === null || (r.item_index ?? 0) === target.item_index);

// only drawn shapes are carried or count as a physician's mark; per-frame Event choices are neither
const REGION_TYPES = new Set([
  "polygonlabels",
  "rectanglelabels",
  "ellipselabels",
  "ellipsebasedpolygonlabels",
  "keypointlabels",
  "brushlabels",
]);
export const isRegion = (r) => REGION_TYPES.has(r?.type);
const regionOn = (r, target) => isRegion(r) && onTarget(r, target);

// a region nobody touched is a proposal; drawn, edited or accepted regions belong to the physician
export const isDoctorRegion = (r) => r.origin !== "prediction";

export const baseId = (id) => String(id ?? "").split("@")[0];

export const acceptTarget = (results, target) =>
  results.map((r) => (regionOn(r, target) && r.origin === "prediction" ? { ...r, origin: "prediction-changed" } : r));

const frameIndex = (toName) => Number.parseInt(String(toName).match(/(\d+)$/)?.[1] ?? "", 10);

export const frameTarget = (frames, target, k) => {
  const i = frames.findIndex((f) => f.to_name === target.to_name);
  const frame = i < 0 ? null : frames[i + k];
  const url = frame?.urls?.[target.item_index ?? 0];
  return typeof url === "string" ? { url, to_name: frame.to_name, item_index: target.item_index } : null;
};

export const mergeProposal = (results, target, proposal) => {
  if (results.some((r) => regionOn(r, target) && isDoctorRegion(r))) return null;
  const idx = frameIndex(target.to_name);
  const where = `${target.to_name}:${target.item_index ?? 0}`;
  const kept = results.filter((r) => !regionOn(r, target));
  const added = proposal.map((r) => ({
    ...r,
    id: `${baseId(r.id)}@${where}`,
    origin: "prediction",
    to_name: target.to_name,
    // controls of a Repeater frame are named <control>-<frame>: point them at the target frame
    ...(r.from_name && Number.isFinite(idx) ? { from_name: String(r.from_name).replace(/\d+$/, String(idx)) } : {}),
    ...(target.item_index === null ? {} : { item_index: target.item_index }),
  }));
  return [...kept, ...added];
};

// one run at a time: a second Shift+Enter while a chain is running is ignored
export const guarded =
  (fn, state = { busy: false }) =>
  async (...args) => {
    if (state.busy) return undefined;
    state.busy = true;
    try {
      return await fn(...args);
    } finally {
      state.busy = false;
    }
  };

/**
 * "Done image": confirm `target`, let the well adapter learn it, propose the next `n` frames.
 * current() -> serialized results; apply(results) writes them; call(image, context) -> LS predict/test response.
 * Throws when a request fails or the backend does not speak the propagation protocol.
 */
export const runChain = async ({ current, apply, call, frames, target, n, auto }) => {
  const carriedFrom = current().filter((r) => regionOn(r, target));
  if (!carriedFrom.length) return { filled: 0, note: "no_regions" };
  apply(acceptTarget(current(), target));
  const gold = { url: target.url, results: carriedFrom };
  const ask = async (image, context) => {
    const resp = await call(image, context);
    if (!resp) throw new Error("yêu cầu tới ML backend thất bại");
    if (resp.errors?.length) throw new Error(resp.errors.join("; "));
    if (!resp.propagation) throw new Error("ML backend này không hỗ trợ lan truyền");
    return resp;
  };
  let carried = carriedFrom;
  let filled = 0;
  let note = "";
  let learned = false;
  for (let k = 1; auto && k <= n; k++) {
    const next = frameTarget(frames, target, k);
    if (!next) {
      note = "đã tới frame cuối";
      break;
    }
    if (mergeProposal(current(), next, []) === null) {
      note = `gặp ảnh đã gán ở ${next.to_name}`;
      break;
    }
    const resp = await ask(next, { carried, distance: k, ...(learned ? {} : { gold }) });
    learned = true;
    const result = (resp.data || resp).result || [];
    // the physician may have drawn on this frame while the request was in flight: never overwrite it
    const merged = mergeProposal(current(), next, result);
    if (!merged) {
      note = `gặp ảnh đã gán ở ${next.to_name}`;
      break;
    }
    apply(merged);
    filled += 1;
    carried = result;
    if (resp.propagation.stop) {
      note = `dừng ở ${next.to_name}: ${resp.propagation.stop}`;
      break;
    }
  }
  if (!learned) await ask(target, { carried: [], distance: 0, gold }); // learn only, propose nothing
  return { filled, note };
};
