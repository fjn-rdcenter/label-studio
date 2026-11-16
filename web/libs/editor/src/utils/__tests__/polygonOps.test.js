import { mergePolygons, signedArea, snapToBoundary, splitPolygon } from "../polygonOps";

const square = (x, y, s) => [
  [x, y],
  [x + s, y],
  [x + s, y + s],
  [x, y + s],
];
const area = (ring) => Math.abs(signedArea(ring));

describe("mergePolygons", () => {
  it("unions overlapping polygons into one", () => {
    const res = mergePolygons([{ points: square(0, 0, 2) }, { points: square(1, 1, 2) }]);

    expect(res.error).toBeUndefined();
    expect(area(res.points)).toBeCloseTo(7);
    expect(res.holes).toEqual([]);
  });

  it("returns the outer polygon when one is inside the other", () => {
    const res = mergePolygons([{ points: square(0, 0, 10) }, { points: square(2, 2, 2) }]);

    expect(area(res.points)).toBeCloseTo(100);
  });

  it("merges polygons that only share an edge", () => {
    const res = mergePolygons([{ points: square(0, 0, 1) }, { points: square(1, 0, 1) }]);

    expect(area(res.points)).toBeCloseTo(2);
  });

  it("refuses disjoint polygons", () => {
    expect(mergePolygons([{ points: square(0, 0, 1) }, { points: square(5, 5, 1) }]).error).toBeTruthy();
  });

  it("keeps a single enclosed hole as a ring", () => {
    const u = [
      [0, 0],
      [3, 0],
      [3, 3],
      [2, 3],
      [2, 1],
      [1, 1],
      [1, 3],
      [0, 3],
    ];
    const res = mergePolygons([
      { points: u },
      {
        points: [
          [0, 2.5],
          [3, 2.5],
          [3, 3.5],
          [0, 3.5],
        ],
      },
    ]);

    expect(res.error).toBeUndefined();
    expect(area(res.holes)).toBeCloseTo(1.5);
  });

  it("refuses results with more than one hole", () => {
    const comb = [
      [0, 0],
      [5, 0],
      [5, 3],
      [4, 3],
      [4, 1],
      [3, 1],
      [3, 3],
      [2, 3],
      [2, 1],
      [1, 1],
      [1, 3],
      [0, 3],
    ];

    expect(
      mergePolygons([
        { points: comb },
        {
          points: [
            [0, 2.5],
            [5, 2.5],
            [5, 3.5],
            [0, 3.5],
          ],
        },
      ]).error,
    ).toBeTruthy();
  });
});

describe("snapToBoundary", () => {
  const ring = square(0, 0, 10);

  it("prefers a nearby vertex", () => {
    expect(snapToBoundary(ring, [0.5, 0.5], 1)).toEqual({ index: 0 });
  });

  it("projects onto an edge", () => {
    expect(snapToBoundary(ring, [5, 0.5], 1)).toEqual({ edge: 0, point: [5, 0] });
  });

  it("returns null away from the boundary", () => {
    expect(snapToBoundary(ring, [5, 5], 1)).toBeNull();
  });
});

describe("splitPolygon", () => {
  const ring = square(0, 0, 10);

  it("splits a square along a diagonal", () => {
    const { pieces, error } = splitPolygon(ring, { index: 0 }, { index: 2 });

    expect(error).toBeUndefined();
    expect(pieces.map(area)).toEqual([50, 50]);
  });

  it("splits from edge to edge through interior points", () => {
    const { pieces } = splitPolygon(ring, { edge: 0, point: [5, 0] }, { edge: 2, point: [5, 10] }, [[3, 5]]);

    expect(pieces).toHaveLength(2);
    expect(area(pieces[0]) + area(pieces[1])).toBeCloseTo(100);
    pieces.forEach((p) => expect(p.length).toBeGreaterThanOrEqual(3));
  });

  it("handles start and end on the same edge", () => {
    const { pieces } = splitPolygon(ring, { edge: 0, point: [7, 0] }, { edge: 0, point: [3, 0] }, [[5, 4]]);

    expect(area(pieces[0]) + area(pieces[1])).toBeCloseTo(100);
    expect(Math.min(...pieces.map(area))).toBeCloseTo(8);
  });

  it("refuses a cut that leaves a concave polygon", () => {
    const l = [
      [0, 0],
      [10, 0],
      [10, 4],
      [4, 4],
      [4, 10],
      [0, 10],
    ];

    expect(splitPolygon(l, { index: 1 }, { index: 5 }).error).toBeTruthy();
    expect(splitPolygon(l, { index: 1 }, { index: 5 }, [[8, 8]]).error).toBeTruthy();
  });

  it("refuses start == end and zero-area pieces", () => {
    expect(splitPolygon(ring, { index: 1 }, { index: 1 }, [[5, 5]]).error).toBeTruthy();
    expect(splitPolygon(ring, { index: 0 }, { index: 1 }).error).toBeTruthy();
  });

  it("refuses a self-crossing cut", () => {
    const res = splitPolygon(ring, { edge: 0, point: [5, 0] }, { edge: 2, point: [5, 10] }, [
      [2, 6],
      [8, 4],
      [8, 6],
      [2, 4],
    ]);

    expect(res.error).toBeTruthy();
  });
});
