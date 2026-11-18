import {
  DEFAULT_SETTINGS,
  acceptTarget,
  frameTarget,
  guarded,
  isDoctorRegion,
  loadSettings,
  mergeProposal,
  runChain,
} from "../propagation";

const t0 = { url: "u0", to_name: "timelapse_0", item_index: 5 };
const frames = [0, 1, 2].map((i) => ({
  to_name: `timelapse_${i}`,
  urls: Array.from({ length: 11 }, (_, s) => `f${i}s${s}`),
}));
const pred = (id, to_name, origin = "prediction") => ({
  id,
  to_name,
  item_index: 5,
  origin,
  type: "polygonlabels",
  value: {},
});

test("loadSettings falls back and clamps", () => {
  expect(
    loadSettings({
      getItem: () => {
        throw new Error("blocked");
      },
    }),
  ).toEqual(DEFAULT_SETTINGS);
  expect(loadSettings({ getItem: () => '{"n":42,"threshold":3,"auto":false}' })).toEqual({
    n: 10,
    threshold: 1,
    auto: false,
  });
});

test("frameTarget moves along frames on the same slice and stops at the end", () => {
  expect(frameTarget(frames, t0, 1)).toEqual({ url: "f1s5", to_name: "timelapse_1", item_index: 5 });
  expect(frameTarget(frames, t0, 3)).toBeNull();
});

test("doctor regions are anchors", () => {
  expect(isDoctorRegion({ origin: "manual" })).toBe(true);
  expect(isDoctorRegion({})).toBe(true);
  expect(isDoctorRegion({ origin: "prediction" })).toBe(false);
  const t1 = frameTarget(frames, t0, 1);
  expect(mergeProposal([pred("a", "timelapse_1", "prediction-changed")], t1, [])).toBeNull();
});

test("mergeProposal replaces only untouched proposals on the target and retargets them", () => {
  const t1 = frameTarget(frames, t0, 1);
  const other = pred("x@timelapse_0:5", "timelapse_0");
  const out = mergeProposal([other, pred("old@timelapse_1:5", "timelapse_1")], t1, [
    { id: "zp1@timelapse_0:5", from_name: "pl-0", to_name: "timelapse_0", value: {} },
  ]);
  expect(out.map((r) => r.id)).toEqual(["x@timelapse_0:5", "zp1@timelapse_1:5"]);
  expect(out[1]).toMatchObject({ origin: "prediction", item_index: 5, to_name: "timelapse_1", from_name: "pl-1" });
});

test("acceptTarget marks the confirmed image as touched", () => {
  const out = acceptTarget([pred("a", "timelapse_0"), pred("b", "timelapse_1")], t0);
  expect(out.map((r) => r.origin)).toEqual(["prediction-changed", "prediction"]);
});

// ---- runChain: the "Done image" loop, with the editor and the API replaced by plain functions ----
const poly = (id, to_name, origin = "manual") => ({
  id,
  to_name,
  item_index: 5,
  origin,
  type: "polygonlabels",
  value: {},
});
const ok = (result, stop = null) => ({ data: { result }, propagation: { stop } });

const chain = (state, call, extra = {}) => {
  const applied = [];
  const deps = {
    current: () => state.results,
    apply: (results) => {
      applied.push(results);
      state.results = results;
    },
    call,
    frames,
    target: { ...t0, url: "f0s5" },
    n: 2,
    auto: true,
    ...extra,
  };
  return { applied, run: () => runChain(deps) };
};

test("a physician region appearing on the target while the request is in flight never wipes the annotation", async () => {
  const state = { results: [poly("a@timelapse_0:5", "timelapse_0")] };
  const call = async () => {
    state.results = [...state.results, poly("mine", "timelapse_1")]; // drawn during the await
    return ok([poly("a", "timelapse_0", "prediction")]);
  };
  const { applied, run } = chain(state, call);
  const out = await run();
  expect(applied.every(Array.isArray)).toBe(true);
  expect(state.results.map((r) => r.id)).toEqual(["a@timelapse_0:5", "mine"]);
  expect(out.note).toMatch(/timelapse_1/);
});

test("a failed request stops the chain with an error instead of an empty proposal", async () => {
  const state = { results: [poly("a", "timelapse_0")] };
  const { run } = chain(state, async () => null);
  await expect(run()).rejects.toThrow();
});

test("a backend that ignores the propagation context is refused", async () => {
  const state = { results: [poly("a", "timelapse_0")] };
  const { run } = chain(state, async () => ({ data: { result: [poly("x", "timelapse_1", "prediction")] } }));
  await expect(run()).rejects.toThrow();
});

test("the confirmed image is still learned when the chain cannot take a single step", async () => {
  const state = { results: [poly("a", "timelapse_0"), poly("b", "timelapse_1")] }; // next frame already labelled
  const calls = [];
  const { run } = chain(state, async (image, context) => {
    calls.push({ image, context });
    return ok([], "no_objects");
  });
  await run();
  expect(calls).toHaveLength(1);
  expect(calls[0].context).toMatchObject({ distance: 0, carried: [], gold: { url: "f0s5" } });
});

test("learn-only mode never proposes onto the confirmed image", async () => {
  const state = { results: [poly("a", "timelapse_0")] };
  const calls = [];
  const { run } = chain(
    state,
    async (image, context) => {
      calls.push(context);
      return ok([], "no_objects");
    },
    { auto: false },
  );
  await run();
  expect(calls).toEqual([expect.objectContaining({ carried: [], distance: 0 })]);
});

test("event choices are neither carried nor anchors", async () => {
  const event = { id: "e", to_name: "timelapse_1", item_index: 5, type: "choices", value: { choices: ["tB"] } };
  const state = { results: [poly("a", "timelapse_0"), event] };
  const calls = [];
  const { run } = chain(
    state,
    async (image, context) => {
      calls.push(context);
      return ok([poly("a", "timelapse_0", "prediction")]);
    },
    { n: 1 },
  );
  await run();
  expect(calls[0].carried.map((r) => r.id)).toEqual(["a"]);
  expect(calls[0].distance).toBe(1);
  expect(state.results.some((r) => r.id === "e")).toBe(true);
});

test("guarded runs one call at a time", async () => {
  let release;
  const fn = guarded(() => new Promise((r) => (release = r)));
  const first = fn();
  expect(await fn()).toBeUndefined();
  release("done");
  expect(await first).toBe("done");
});
