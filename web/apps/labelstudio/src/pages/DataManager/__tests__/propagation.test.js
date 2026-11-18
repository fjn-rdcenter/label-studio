import {
  DEFAULT_SETTINGS,
  acceptTarget,
  frameTarget,
  isDoctorRegion,
  loadSettings,
  mergeProposal,
} from "../propagation";

const t0 = { url: "u0", to_name: "timelapse_0", item_index: 5 };
const frames = [0, 1, 2].map((i) => ({
  to_name: `timelapse_${i}`,
  urls: Array.from({ length: 11 }, (_, s) => `f${i}s${s}`),
}));
const pred = (id, to_name, origin = "prediction") => ({ id, to_name, item_index: 5, origin, value: {} });

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
