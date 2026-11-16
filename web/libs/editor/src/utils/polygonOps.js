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

/**
 * Union polygons that touch or overlap into one.
 * @param {{points: number[][], holes?: number[][]}[]} polygons
 * @returns {{points: number[][], holes: number[][]} | {error: string}}
 */
export function mergePolygons(polygons) {
  const result = polygonClipping.union(...polygons.map(toGeom));

  if (result.length !== 1) return { error: "Các polygon phải chạm hoặc chồng lên nhau mới gộp được." };

  return fromMulti(result)?.[0] ?? { error: TOO_MANY_HOLES };
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
