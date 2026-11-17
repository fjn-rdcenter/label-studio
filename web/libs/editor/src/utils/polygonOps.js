import polygonClipping from "polygon-clipping";

// Pure geometry for the polygon Merge/Cut tools.
// A ring is an open array of [x, y] points (last point != first point).
// A polygon is {points: ring, holes: ring} — PolygonRegion stores at most one hole contour.

const openRing = (ring) => ring.slice(0, -1); // polygon-clipping returns closed rings

const dist2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;

const toGeom = ({ points, holes }) => (holes?.length ? [points, holes] : [points]);

// polygon-clipping MultiPolygon -> polygons. PolygonRegion holds one hole, so a piece with
// several keeps its largest one and the smaller ones are filled; filled counts them.
const fromMulti = (multi) => ({
  polygons: multi.map(([outer, ...holes]) => {
    if (!holes.length) return { points: openRing(outer), holes: [] };
    const hole = holes.reduce((a, b) => (Math.abs(signedArea(b)) > Math.abs(signedArea(a)) ? b : a));

    return { points: openRing(outer), holes: openRing(hole) };
  }),
  filled: multi.reduce((n, [, ...holes]) => n + Math.max(0, holes.length - 1), 0),
});

export const signedArea = (ring) =>
  ring.reduce((sum, p, i) => {
    const q = ring[(i + 1) % ring.length];

    return sum + p[0] * q[1] - q[0] * p[1];
  }, 0) / 2;

const polyArea = ({ points, holes }) =>
  Math.abs(signedArea(points)) - (holes?.length ? Math.abs(signedArea(holes)) : 0);

const multiArea = (multi) =>
  multi.reduce(
    (sum, [outer, ...holes]) =>
      sum + Math.abs(signedArea(outer)) - holes.reduce((h, r) => h + Math.abs(signedArea(r)), 0),
    0,
  );

const distToSegment = (p, a, b) => {
  const len2 = dist2(a, b);
  const t = len2 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / len2)) : 0;

  return Math.sqrt(dist2(p, [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]));
};

const distToRing = (p, ring) => Math.min(...ring.map((a, i) => distToSegment(p, a, ring[(i + 1) % ring.length])));

// Drop vertices that don't belong to `big` and sit within tol of its contours, as long as
// removing them moves the outline by at most tol, so a small polygon blends into the big one.
function absorbVertices(ring, big, tol) {
  const own = new Set(big.points.concat(big.holes ?? []).map((p) => p.join()));
  const contours = [big.points, big.holes].filter((r) => r?.length);
  const out = [...ring];

  for (let removed = true; removed && out.length > 3; ) {
    removed = false;
    for (let i = 0; i < out.length && out.length > 3; i++) {
      const v = out[i];
      const prev = out[(i + out.length - 1) % out.length];
      const next = out[(i + 1) % out.length];

      if (own.has(v.join())) continue;
      if (Math.min(...contours.map((r) => distToRing(v, r))) > tol || distToSegment(v, prev, next) > tol) continue;
      out.splice(i--, 1);
      removed = true;
    }
  }
  return out;
}

// Of the union's holes keep the largest one that lies mostly in a hole of the inputs (e.g. a ring's
// inner contour); holes the merge itself created are gaps between the pieces and get filled.
function inheritedHole(holes, inputs) {
  const inputHoles = inputs.filter((p) => p.holes?.length).map((p) => [p.holes]);

  if (!holes.length || !inputHoles.length) return [];
  const covered = polygonClipping.union(...inputHoles);
  const kept = holes.filter(
    (h) => multiArea(polygonClipping.intersection([h], covered)) >= Math.abs(signedArea(h)) / 2,
  );

  if (!kept.length) return [];
  return openRing(kept.reduce((a, b) => (Math.abs(signedArea(b)) > Math.abs(signedArea(a)) ? b : a)));
}

// Close slits: a vertex within eps of a non-neighbouring edge cuts the outline in two loops; the
// loop running the opposite way encloses outside space (a crack between merged pieces), so it is
// dropped and the outline jumps straight across. Loops running the same way are real thin parts.
export function closeSlits(ring, eps) {
  let out = [...ring];
  const sign = Math.sign(signedArea(out));
  const opposite = (loop) => Math.sign(signedArea(loop)) === -sign && Math.abs(signedArea(loop)) > 1e-12;

  for (let changed = true; changed && out.length > 3; ) {
    changed = false;
    const n = out.length;

    for (let i = 0; i < n && !changed; i++) {
      // edges (j, j+1) that share no vertex with the edges around i
      for (let k = 2; k <= n - 3 && !changed; k++) {
        const j = (i + k) % n;
        const [a, b] = [out[j], out[(j + 1) % n]];

        if (distToSegment(out[i], a, b) > eps) continue;
        const len2 = dist2(a, b);
        const t = len2
          ? Math.max(0, Math.min(1, ((out[i][0] - a[0]) * (b[0] - a[0]) + (out[i][1] - a[1]) * (b[1] - a[1])) / len2))
          : 0;
        const q = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
        const arc = (from, to) => Array.from({ length: ((to - from + n) % n) + 1 }, (_, s) => out[(from + s) % n]);
        const loopA = [...arc(i, j), q]; // i forward to j, then across to q
        const loopB = [q, ...arc((j + 1) % n, i)]; // q, j+1 forward to i

        if (opposite(loopA)) out = loopB;
        else if (opposite(loopB)) out = loopA;
        else continue;
        out = out.filter((p, s) => dist2(p, out[(s + 1) % out.length]) > 0);
        changed = true;
      }
    }
  }
  return out;
}

// Drop vertices that barely shape the outline: one of two neighbours closer than eps/2, or a vertex
// whose triangle with its neighbours is under eps²/2 (1px zigzags, sliver spikes, collinear points).
// Smallest triangle first, so real corners (large triangles) stay; at least 3 vertices remain.
export function dropTinyVertices(ring, eps) {
  const out = [...ring];
  const minArea = (eps * eps) / 2;
  const tri = (a, b, c) => Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2;
  const close = (a, b) => dist2(a, b) < (eps / 2) ** 2;

  while (out.length > 3) {
    const n = out.length;
    let best = -1;
    let bestArea = Number.POSITIVE_INFINITY;

    out.forEach((p, i) => {
      const prev = out[(i - 1 + n) % n];
      const next = out[(i + 1) % n];
      const a = tri(prev, p, next);

      if ((a < minArea || close(p, prev) || close(p, next)) && a < bestArea) [best, bestArea] = [i, a];
    });
    if (best < 0) break;
    out.splice(best, 1);
  }
  return out;
}

const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

// Convex hull (Andrew's monotone chain), counter-clockwise, no collinear points
export function convexHull(points) {
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const half = (list) =>
    list.reduce((h, p) => {
      while (h.length > 1 && cross(h[h.length - 2], h[h.length - 1], p) <= 0) h.pop();
      h.push(p);
      return h;
    }, []);
  const lower = half(pts);
  const upper = half([...pts].reverse());

  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

/**
 * Union polygons into one. Polygons that touch or overlap are unioned; if they end up in
 * separate pieces, the result is the convex hull of all of them (one big shape, no hole).
 * @param {{points: number[][], holes?: number[][]}[]} polygons
 * Holes the merge creates (gaps between the pieces) are filled; only an input's own hole is kept.
 * @param {number} tol with tol > 0, vertices of the smaller polygons lying within tol of the
 *   largest one are dropped so they blend into it (hole contour of the result is left as is)
 * @param {number} eps with eps > 0, slits narrower than eps left between the pieces are closed and
 *   vertices that barely shape the outline (near duplicates, 1px zigzags, sliver spikes) are dropped
 * @returns {{points: number[][], holes: number[][], bigIndex: number, absorbed: boolean} | {error: string}}
 *   bigIndex: the largest polygon; absorbed: every other one lies at least half inside it
 */
export function mergePolygons(polygons, tol = 0, eps = 0) {
  const result = polygonClipping.union(...polygons.map(toGeom));
  let merged = { points: convexHull(polygons.flatMap((p) => p.points)), holes: [] };

  if (result.length === 1) {
    const [outer, ...holes] = result[0];

    merged = {
      points: eps > 0 ? dropTinyVertices(closeSlits(openRing(outer), eps), eps) : openRing(outer),
      holes: inheritedHole(holes, polygons),
    };
  }
  const areas = polygons.map(polyArea);
  const bigIndex = areas.indexOf(Math.max(...areas));
  const big = polygons[bigIndex];
  const absorbed = polygons.every(
    (p, i) => i === bigIndex || multiArea(polygonClipping.intersection(toGeom(p), toGeom(big))) >= areas[i] / 2,
  );
  const points = tol > 0 ? absorbVertices(merged.points, big, tol) : merged.points;

  return { ...merged, points, bigIndex, absorbed };
}

/**
 * Cut a drawn closed area out of a polygon.
 * @param {{points: number[][], holes?: number[][]}} polygon
 * @param {number[][]} area ring drawn by the user, may extend outside the polygon
 * @returns {{inside: object[], outside: object[], filledHoles: number} | {error: string}} inside = new polygons,
 *   outside = what is left, filledHoles = holes dropped because a polygon keeps only its largest one
 */
export function cutPolygon(polygon, area) {
  const geom = toGeom(polygon);
  const inside = fromMulti(polygonClipping.intersection(geom, [area]));
  const outside = fromMulti(polygonClipping.difference(geom, [area]));

  if (!inside.polygons.length) return { error: "Vùng cắt không giao với polygon." };
  if (!outside.polygons.length) return { error: "Vùng cắt phủ hết polygon, không còn gì để tách." };
  return { inside: inside.polygons, outside: outside.polygons, filledHoles: inside.filled + outside.filled };
}

/**
 * Carve the other polygons out of the one with the largest outline (holes counted as covered,
 * so a ring wins over a solid polygon sitting in its hole). The carving polygons stay as they are.
 * @param {{points: number[][], holes?: number[][]}[]} polygons
 * @returns {{outerIndex: number, pieces: object[], filledHoles: number} | {error: string}}
 *   pieces replace polygons[outerIndex]; filledHoles as in cutPolygon
 */
export function carvePolygons(polygons) {
  const outlines = polygons.map((p) => Math.abs(signedArea(p.points)));
  const outerIndex = outlines.indexOf(Math.max(...outlines));
  const outer = toGeom(polygons[outerIndex]);
  const others = polygons.filter((_, i) => i !== outerIndex).map(toGeom);

  if (!others.some((g) => multiArea(polygonClipping.intersection(outer, g)) > 0)) {
    return { error: "Các polygon không chồng lên nhau." };
  }

  const { polygons: pieces, filled } = fromMulti(polygonClipping.difference(outer, ...others));

  if (!pieces.length) return { error: "Polygon lớn bị khoét hết." };
  return { outerIndex, pieces, filledHoles: filled };
}

/**
 * Snap a point to the nearest vertex (preferred) or edge of the given rings.
 * @returns {{ring: number, edge: number, t: number, point: number[]} | null}
 *   ring index, edge i (from ring[i] to ring[i + 1]) and position t along it
 */
export function snapToContour(rings, pt, tol) {
  const tol2 = tol * tol;
  let vertex = null;
  let edge = null;

  rings.forEach((ring, k) => {
    ring.forEach((a, i) => {
      const d = dist2(a, pt);

      if (d <= tol2 && (!vertex || d < vertex.d)) vertex = { ring: k, edge: i, t: 0, point: a, d };

      const b = ring[(i + 1) % ring.length];
      const len2 = dist2(a, b);

      if (!len2) return;
      const t = Math.max(0, Math.min(1, ((pt[0] - a[0]) * (b[0] - a[0]) + (pt[1] - a[1]) * (b[1] - a[1])) / len2));
      const point = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
      const e = dist2(point, pt);

      if (e <= tol2 && (!edge || e < edge.d)) edge = { ring: k, edge: i, t, point, d: e };
    });
  });

  const best = vertex ?? edge;

  return best && { ring: best.ring, edge: best.edge, t: best.t, point: best.point };
}

// Vertices met walking forward along the ring from snap a to snap b, ends excluded
function contourBetween(ring, a, b) {
  if (a.edge === b.edge && a.t <= b.t) return [];
  const out = [];

  for (let i = (a.edge + 1) % ring.length; ; i = (i + 1) % ring.length) {
    if (i === b.edge) return b.t > 0 ? [...out, ring[i]] : out;
    out.push(ring[i]);
  }
}

/**
 * Cut a polygon along an open line from one contour point to another on the same contour.
 * The line is closed by walking that contour back from end to start, then cut like a drawn area.
 * @param {number[][][]} rings contours of the polygon: [outer, hole?]
 * @param {object} start snap from snapToContour
 * @param {object} end snap from snapToContour, on the same ring as start
 * @param {number[][]} line points clicked between start and end
 */
export function cutAlongLine(polygon, rings, start, end, line = []) {
  const area = [start.point, ...line, end.point, ...contourBetween(rings[start.ring], end, start)];

  return cutPolygon(polygon, area);
}

// Indexes of the points inside the box spanned by corners a and b (any order, edges included)
export function pointsInRect(points, a, b) {
  const [x0, x1] = [Math.min(a[0], b[0]), Math.max(a[0], b[0])];
  const [y0, y1] = [Math.min(a[1], b[1]), Math.max(a[1], b[1])];

  return points.flatMap(([x, y], i) => (x >= x0 && x <= x1 && y >= y0 && y <= y1 ? [i] : []));
}

// Shrink a move (dx, dy) so every point stays inside [0, w] x [0, h]
export function clampDelta(points, dx, dy, w, h) {
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);

  return [
    Math.max(-Math.min(...xs), Math.min(w - Math.max(...xs), dx)),
    Math.max(-Math.min(...ys), Math.min(h - Math.max(...ys), dy)),
  ];
}

// Serialized polygon value with new geometry, as PolygonRegion.serialize writes rings
export function withGeometry(value, { points, holes }) {
  const { ring, outerclosed, holes: _, ...rest } = value;

  return holes.length ? { ...rest, points, ring: true, outerclosed: true, holes: [holes] } : { ...rest, points };
}
