import {
  carvePolygons,
  convexHull,
  cutAlongLine,
  cutPolygon,
  mergePolygons,
  signedArea,
  snapToContour,
  withGeometry,
} from "../polygonOps";

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

  it("joins disjoint polygons into their convex hull", () => {
    const res = mergePolygons([{ points: square(0, 0, 1) }, { points: square(5, 5, 1) }], 0.1);

    expect(res.error).toBeUndefined();
    expect(res.points).toHaveLength(6);
    expect(area(res.points)).toBeCloseTo(11);
    expect(res.holes).toEqual([]);
    expect(res.absorbed).toBe(false);
  });

  it("convexHull drops interior and collinear points", () => {
    const hull = convexHull([...square(0, 0, 4), [2, 2], [2, 0], [1, 3]]);

    expect(hull).toHaveLength(4);
    expect(area(hull)).toBeCloseTo(16);
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

describe("mergePolygons absorbing a small polygon", () => {
  const big = { points: square(0, 0, 10), holes: [] };
  // small polygon along the right edge, mostly inside, sticking out by 1
  const hugging = {
    points: [
      [8, 3],
      [11, 3],
      [11, 7],
      [8, 7],
    ],
    holes: [],
  };
  // small polygon sticking out by 10
  const bump = {
    points: [
      [8, 3],
      [20, 3],
      [20, 7],
      [8, 7],
    ],
    holes: [],
  };

  it("drops the small polygon's vertices near the big outline", () => {
    const res = mergePolygons([hugging, big], 2);

    expect(res.points).toHaveLength(4);
    expect(area(res.points)).toBeCloseTo(100);
  });

  it("keeps a clear protrusion", () => {
    const res = mergePolygons([big, bump], 2);

    expect(area(res.points)).toBeCloseTo(100 + 10 * 4);
    expect(res.points).toEqual(
      expect.arrayContaining([
        [20, 3],
        [20, 7],
      ]),
    );
  });

  it("keeps the union as is without a tolerance", () => {
    expect(mergePolygons([hugging, big]).points).toHaveLength(8);
  });

  it("reports the largest polygon even when it is not first", () => {
    expect(mergePolygons([hugging, big], 2).bigIndex).toBe(1);
  });

  it("flags absorbed when the small polygon lies mostly inside", () => {
    expect(mergePolygons([big, hugging]).absorbed).toBe(true);
    // corner overlap only: 1 of 16
    expect(mergePolygons([big, { points: square(9, 9, 4), holes: [] }]).absorbed).toBe(false);
  });
});

describe("carvePolygons", () => {
  const net = (p) => area(p.points) - (p.holes.length ? area(p.holes) : 0);

  it("puts a hole in the big polygon where the small one sits", () => {
    const res = carvePolygons([
      { points: square(4, 4, 2), holes: [] },
      { points: square(0, 0, 10), holes: [] },
    ]);

    expect(res.outerIndex).toBe(1);
    expect(res.pieces).toHaveLength(1);
    expect(area(res.pieces[0].holes)).toBeCloseTo(4);
    expect(net(res.pieces[0])).toBeCloseTo(96);
  });

  it("widens a ring's hole by the polygon covering it (TE/BC), picking the ring by its outline", () => {
    // thin ring: solid area 36 < 81 of the carving polygon, outline 100 > 81
    const ring = { points: square(0, 0, 10), holes: square(1, 1, 8) };
    const res = carvePolygons([ring, { points: square(0.5, 0.5, 9), holes: [] }]);

    expect(res.outerIndex).toBe(0);
    expect(res.pieces).toHaveLength(1);
    expect(area(res.pieces[0].holes)).toBeCloseTo(81);
    expect(net(res.pieces[0])).toBeCloseTo(19);
  });

  it("splits the big polygon when the small one crosses it", () => {
    const res = carvePolygons([
      { points: square(0, 0, 10), holes: [] },
      {
        points: [
          [4, -1],
          [6, -1],
          [6, 11],
          [4, 11],
        ],
        holes: [],
      },
    ]);

    expect(res.pieces).toHaveLength(2);
    expect(res.pieces.reduce((a, p) => a + net(p), 0)).toBeCloseTo(80);
  });

  it("refuses polygons that don't overlap and results with two holes", () => {
    expect(
      carvePolygons([
        { points: square(0, 0, 2), holes: [] },
        { points: square(5, 5, 1), holes: [] },
      ]).error,
    ).toBeTruthy();
    expect(
      carvePolygons([
        { points: square(0, 0, 10), holes: square(1, 1, 2) },
        { points: square(6, 6, 2), holes: [] },
      ]).error,
    ).toBeTruthy();
  });
});

describe("snapToContour", () => {
  const ring = square(0, 0, 10);
  const hole = square(4, 4, 2);

  it("prefers a nearby vertex", () => {
    expect(snapToContour([ring], [0.5, 0.5], 1)).toEqual({ ring: 0, edge: 0, t: 0, point: [0, 0] });
  });

  it("projects onto an edge", () => {
    expect(snapToContour([ring], [5, 0.5], 1)).toEqual({ ring: 0, edge: 0, t: 0.5, point: [5, 0] });
  });

  it("snaps to the hole contour too", () => {
    expect(snapToContour([ring, hole], [4.2, 5], 0.5)).toMatchObject({ ring: 1, point: [4, 5] });
  });

  it("returns null away from every contour", () => {
    expect(snapToContour([ring, hole], [2, 2], 1)).toBeNull();
  });
});

describe("cutAlongLine", () => {
  const sq = { points: square(0, 0, 10), holes: [] };
  const snap = (rings, pt) => snapToContour(rings, pt, 0.5);
  const total = (res) => [...res.inside, ...res.outside].reduce((a, p) => a + area(p.points) - area(p.holes), 0);

  it("halves a square from the top edge to the bottom edge", () => {
    const rings = [sq.points];
    const res = cutAlongLine(sq, rings, snap(rings, [5, 0]), snap(rings, [5, 10]));

    expect(res.error).toBeUndefined();
    expect([...res.inside, ...res.outside].map((p) => area(p.points))).toEqual([50, 50]);
  });

  it("cuts vertex to vertex through an interior point", () => {
    const rings = [sq.points];
    const res = cutAlongLine(sq, rings, snap(rings, [0, 0]), snap(rings, [10, 10]), [[3, 7]]);

    expect(res.inside).toHaveLength(1);
    expect(res.outside).toHaveLength(1);
    expect(total(res)).toBeCloseTo(100);
  });

  it("cuts a notch when start and end sit on the same edge", () => {
    const rings = [sq.points];
    const res = cutAlongLine(sq, rings, snap(rings, [7, 0]), snap(rings, [3, 0]), [[5, 4]]);

    expect(Math.min(...[...res.inside, ...res.outside].map((p) => area(p.points)))).toBeCloseTo(8);
    expect(total(res)).toBeCloseTo(100);
  });

  it("halves a ring by crossing through its hole", () => {
    const ringPoly = { points: square(0, 0, 10), holes: square(3, 3, 4) };
    const rings = [ringPoly.points, ringPoly.holes];
    const res = cutAlongLine(ringPoly, rings, snap(rings, [5, 0]), snap(rings, [5, 10]), [
      [5, 3],
      [5, 7],
    ]);

    expect(res.error).toBeUndefined();
    const pieces = [...res.inside, ...res.outside];

    expect(pieces).toHaveLength(2);
    pieces.forEach((p) => {
      expect(p.holes).toEqual([]);
      expect(area(p.points)).toBeCloseTo(42);
    });
  });

  it("keeps the total area when the line leaves a concave polygon", () => {
    const l = {
      points: [
        [0, 0],
        [10, 0],
        [10, 4],
        [4, 4],
        [4, 10],
        [0, 10],
      ],
      holes: [],
    };
    const rings = [l.points];
    const res = cutAlongLine(l, rings, snap(rings, [10, 2]), snap(rings, [2, 10]), [[8, 8]]);

    expect(res.error).toBeUndefined();
    expect(total(res)).toBeCloseTo(64);
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
