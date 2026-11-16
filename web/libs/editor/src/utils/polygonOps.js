import polygonClipping from "polygon-clipping";

// Pure geometry for the polygon Merge/Split tools.
// A ring is an open array of [x, y] points (last point != first point).

const openRing = (ring) => ring.slice(0, -1); // polygon-clipping returns closed rings

const dist2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;

export const signedArea = (ring) =>
  ring.reduce((sum, p, i) => {
    const q = ring[(i + 1) % ring.length];

    return sum + p[0] * q[1] - q[0] * p[1];
  }, 0) / 2;

export const pointInPolygon = ([x, y], ring) => {
  let inside = false;

  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];

    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
};

const orient = (a, b, c) => Math.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));

// Proper crossing only: touching at an endpoint or collinear overlap does not count.
export const segmentsCross = (a, b, c, d) => {
  const o1 = orient(a, b, c);
  const o2 = orient(a, b, d);
  const o3 = orient(c, d, a);
  const o4 = orient(c, d, b);

  return o1 * o2 < 0 && o3 * o4 < 0;
};

/**
 * Union polygons that touch or overlap into one.
 * @param {{points: number[][], holes?: number[][]}[]} polygons
 * @returns {{points: number[][], holes: number[][]} | {error: string}}
 */
export function mergePolygons(polygons) {
  const geoms = polygons.map(({ points, holes }) => (holes?.length ? [points, holes] : [points]));
  const result = polygonClipping.union(...geoms);

  if (result.length !== 1) return { error: "Các polygon phải chạm hoặc chồng lên nhau mới gộp được." };

  const [outer, ...holes] = result[0];

  // PolygonRegion stores a single hole contour
  if (holes.length > 1) return { error: "Kết quả gộp có nhiều hơn một lỗ, polygon không hỗ trợ." };

  return { points: openRing(outer), holes: holes.length ? openRing(holes[0]) : [] };
}

/**
 * Snap a point to the polygon boundary.
 * @returns {{index: number} | {edge: number, point: number[]} | null}
 */
export function snapToBoundary(ring, pt, tol) {
  const tol2 = tol * tol;
  let best = null;

  ring.forEach((p, index) => {
    const d = dist2(p, pt);

    if (d <= tol2 && (!best || d < best.d)) best = { index, d };
  });
  if (best) return { index: best.index };

  ring.forEach((a, edge) => {
    const b = ring[(edge + 1) % ring.length];
    const len2 = dist2(a, b);

    if (!len2) return;
    const t = Math.max(0, Math.min(1, ((pt[0] - a[0]) * (b[0] - a[0]) + (pt[1] - a[1]) * (b[1] - a[1])) / len2));
    const point = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
    const d = dist2(point, pt);

    if (d <= tol2 && (!best || d < best.d)) best = { edge, point, d };
  });
  return best && { edge: best.edge, point: best.point };
}

// Insert edge snaps into the ring; returns the new ring and each snap's index in it.
function insertSnaps(ring, snaps) {
  const out = [];
  const idx = snaps.map(() => -1);

  ring.forEach((p, i) => {
    snaps.forEach((s, k) => {
      if (s.index === i) idx[k] = out.length;
    });
    out.push(p);
    snaps
      .map((s, k) => [s, k])
      .filter(([s]) => s.edge === i)
      .sort(([a], [b]) => dist2(p, a.point) - dist2(p, b.point))
      .forEach(([s, k]) => {
        idx[k] = out.length;
        out.push(s.point);
      });
  });
  return { ring: out, idx };
}

/**
 * Split a polygon in two along a cut line that starts and ends on its boundary.
 * @param {number[][]} ring polygon outline
 * @param {object} start snap from snapToBoundary
 * @param {object} end snap from snapToBoundary
 * @param {number[][]} cut interior points from start to end
 * @returns {{pieces: number[][][]} | {error: string}}
 */
export function splitPolygon(ring, start, end, cut = []) {
  const { ring: full, idx } = insertSnaps(ring, [start, end]);
  const [i, j] = idx;
  const n = full.length;

  if (i === j || dist2(full[i], full[j]) === 0) return { error: "Điểm đầu và điểm cuối đường cắt trùng nhau." };

  const outside = { error: "Đường cắt phải nằm bên trong polygon." };

  if (cut.some((p) => !pointInPolygon(p, ring))) return outside;

  const line = [full[i], ...cut, full[j]];

  for (let s = 0; s < line.length - 1; s++) {
    const [a, b] = [line[s], line[s + 1]];

    // midpoint test catches a chord that leaves a concave polygon through a vertex
    if (!pointInPolygon([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], ring)) return outside;
    for (let e = 0; e < n; e++) {
      if (segmentsCross(a, b, full[e], full[(e + 1) % n])) return { error: "Đường cắt cắt ngang biên polygon." };
    }
    for (let t = s + 2; t < line.length - 1; t++) {
      if (segmentsCross(a, b, line[t], line[t + 1])) return { error: "Đường cắt tự cắt chính nó." };
    }
  }

  const walk = (from, to) => {
    const res = [];

    for (let k = from; ; k = (k + 1) % n) {
      res.push(full[k]);
      if (k === to) return res;
    }
  };
  const pieces = [
    [...walk(i, j), ...[...cut].reverse()],
    [...walk(j, i), ...cut],
  ];
  const minArea = Math.abs(signedArea(ring)) * 1e-6;

  if (pieces.some((p) => p.length < 3 || Math.abs(signedArea(p)) <= minArea)) {
    return { error: "Đường cắt tạo ra mảnh rỗng." };
  }
  return { pieces };
}
