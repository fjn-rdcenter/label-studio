import { cutPolygon, mergePolygons, signedArea, snapPoint, withGeometry } from "../polygonOps";

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

describe("snapPoint", () => {
  const ring = square(0, 0, 10);
  const hole = square(4, 4, 2);

  it("prefers a nearby vertex", () => {
    expect(snapPoint([ring], [0.5, 0.5], 1)).toEqual([0, 0]);
  });

  it("projects onto an edge", () => {
    expect(snapPoint([ring], [5, 0.5], 1)).toEqual([5, 0]);
  });

  it("snaps to the hole contour too", () => {
    expect(snapPoint([ring, hole], [4.2, 5], 0.5)).toEqual([4, 5]);
  });

  it("returns null away from every contour", () => {
    expect(snapPoint([ring, hole], [2, 2], 1)).toBeNull();
  });
});

describe("cutPolygon", () => {
  const sq = { points: square(0, 0, 10), holes: [] };

  it("splits a square with an area drawn past its edge", () => {
    const res = cutPolygon(sq, [
      [5, -2],
      [12, -2],
      [12, 12],
      [5, 12],
    ]);

    expect(res.error).toBeUndefined();
    expect(res.inside.map((p) => area(p.points))).toEqual([50]);
    expect(res.outside.map((p) => area(p.points))).toEqual([50]);
  });

  it("carves a hole when the area is strictly inside", () => {
    const res = cutPolygon(sq, square(4, 4, 2));

    expect(area(res.inside[0].points)).toBeCloseTo(4);
    expect(area(res.outside[0].points)).toBeCloseTo(100);
    expect(area(res.outside[0].holes)).toBeCloseTo(4);
  });

  it("cuts a slice out of a ring (annulus) leaving a C shape", () => {
    const ring = { points: square(0, 0, 10), holes: square(3, 3, 4) };
    const res = cutPolygon(ring, [
      [4, -1],
      [6, -1],
      [6, 5],
      [4, 5],
    ]);

    expect(res.error).toBeUndefined();
    expect(res.inside).toHaveLength(1);
    expect(area(res.inside[0].points)).toBeCloseTo(6);
    expect(res.outside).toHaveLength(1);
    expect(res.outside[0].holes).toEqual([]);
    expect(area(res.outside[0].points)).toBeCloseTo(100 - 16 - 6);
  });

  it("splits a ring in two with an area across both sides", () => {
    const ring = { points: square(0, 0, 10), holes: square(3, 3, 4) };
    const res = cutPolygon(ring, [
      [-1, -1],
      [5, -1],
      [5, 11],
      [-1, 11],
    ]);

    expect(res.inside).toHaveLength(1);
    expect(res.outside).toHaveLength(1);
    expect(area(res.inside[0].points) + area(res.outside[0].points)).toBeCloseTo(84);
  });

  it("refuses an area that misses or covers the polygon", () => {
    expect(cutPolygon(sq, square(20, 20, 2)).error).toBeTruthy();
    expect(cutPolygon(sq, square(-1, -1, 12)).error).toBeTruthy();
  });

  it("refuses results with two holes", () => {
    const withHole = { points: square(0, 0, 10), holes: square(1, 1, 2) };

    expect(cutPolygon(withHole, square(6, 6, 2)).error).toBeTruthy();
  });
});

describe("withGeometry", () => {
  const value = { points: [], polygonlabels: ["ZP"], closed: true, ring: true, outerclosed: true, holes: [[[1, 1]]] };

  it("drops ring fields for a polygon without a hole", () => {
    expect(withGeometry(value, { points: [[0, 0]], holes: [] })).toEqual({
      points: [[0, 0]],
      polygonlabels: ["ZP"],
      closed: true,
    });
  });

  it("writes a single nested hole like PolygonRegion.serialize", () => {
    expect(withGeometry(value, { points: [[0, 0]], holes: [[2, 2]] })).toMatchObject({
      ring: true,
      outerclosed: true,
      holes: [[[2, 2]]],
    });
  });
});
