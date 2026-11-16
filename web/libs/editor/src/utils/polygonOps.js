import polygonClipping from "polygon-clipping";

// Pure geometry for the polygon Merge/Cut tools.
// A ring is an open array of [x, y] points (last point != first point).
// A polygon is {points: ring, holes: ring} — PolygonRegion stores at most one hole contour.

const TOO_MANY_HOLES = "Kết quả có nhiều hơn một lỗ, polygon không hỗ trợ.";

const openRing = (ring) => ring.slice(0, -1); // polygon-clipping returns closed rings

const dist2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;

const toGeom = ({ points, holes }) => (holes?.length ? [points, holes] : [points]);

// polygon-clipping MultiPolygon -> polygons, or null if one has more than one hole
const fromMulti = (multi) =>
  multi.some((poly) => poly.length > 2)
    ? null
    : multi.map(([outer, hole]) => ({ points: openRing(outer), holes: hole ? openRing(hole) : [] }));

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

/**
 * Union polygons that touch or overlap into one.
 * @param {{points: number[][], holes?: number[][]}[]} polygons
 * @param {number} tol with tol > 0, vertices of the smaller polygons lying within tol of the
 *   largest one are dropped so they blend into it (hole contour of the result is left as is)
 * @returns {{points: number[][], holes: number[][], bigIndex: number, absorbed: boolean} | {error: string}}
 *   bigIndex: the largest polygon; absorbed: every other one lies at least half inside it
 */
export function mergePolygons(polygons, tol = 0) {
  const result = polygonClipping.union(...polygons.map(toGeom));

  if (result.length !== 1) return { error: "Các polygon phải chạm hoặc chồng lên nhau mới gộp được." };

  const merged = fromMulti(result)?.[0];

  if (!merged) return { error: TOO_MANY_HOLES };

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
 * @returns {{inside: object[], outside: object[]} | {error: string}} inside = new polygons, outside = what is left
 */
export function cutPolygon(polygon, area) {
  const geom = toGeom(polygon);
  const inside = fromMulti(polygonClipping.intersection(geom, [area]));
  const outside = fromMulti(polygonClipping.difference(geom, [area]));

  if (!inside || !outside) return { error: TOO_MANY_HOLES };
  if (!inside.length) return { error: "Vùng cắt không giao với polygon." };
  if (!outside.length) return { error: "Vùng cắt phủ hết polygon, không còn gì để tách." };
  return { inside, outside };
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

// Serialized polygon value with new geometry, as PolygonRegion.serialize writes rings
export function withGeometry(value, { points, holes }) {
  const { ring, outerclosed, holes: _, ...rest } = value;

  return holes.length ? { ...rest, points, ring: true, outerclosed: true, holes: [holes] } : { ...rest, points };
}
