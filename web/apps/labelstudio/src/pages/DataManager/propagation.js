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

// a region nobody touched is a proposal; drawn, edited or accepted regions belong to the physician
export const isDoctorRegion = (r) => r.origin !== "prediction";

export const baseId = (id) => String(id ?? "").split("@")[0];

export const acceptTarget = (results, target) =>
  results.map((r) => (onTarget(r, target) && r.origin === "prediction" ? { ...r, origin: "prediction-changed" } : r));

const frameIndex = (toName) => Number.parseInt(String(toName).match(/(\d+)$/)?.[1] ?? "", 10);

export const frameTarget = (frames, target, k) => {
  const i = frames.findIndex((f) => f.to_name === target.to_name);
  const frame = i < 0 ? null : frames[i + k];
  const url = frame?.urls?.[target.item_index ?? 0];
  return typeof url === "string" ? { url, to_name: frame.to_name, item_index: target.item_index } : null;
};

export const mergeProposal = (results, target, proposal) => {
  if (results.some((r) => onTarget(r, target) && isDoctorRegion(r))) return null;
  const idx = frameIndex(target.to_name);
  const where = `${target.to_name}:${target.item_index ?? 0}`;
  const kept = results.filter((r) => !onTarget(r, target));
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
