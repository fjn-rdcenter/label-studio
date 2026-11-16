import Konva from "konva";
import { observer } from "mobx-react";
import { types } from "mobx-state-tree";

import BaseTool from "./Base";
import ToolMixin from "../mixins/Tool";
import { Tool } from "../components/Toolbar/Tool";
import InfoModal from "../components/Infomodal/Infomodal";
import { IconSplitTool } from "../assets/icons";
import { carvePolygons, cutAlongLine, cutPolygon, snapToContour, withGeometry } from "../utils/polygonOps";
import { selectedPolygons } from "./PolygonMerge";

// how close (screen px) a click must be to snap to the polygon or to close the drawn area
const SNAP_PX = 8;

const toCanvas = (points) => points.map((p) => [p.canvasX, p.canvasY]);
const regionGeometry = (r) => ({ points: toCanvas(r.points), holes: toCanvas(r.holes) });

const ToolView = observer(({ item }) => (
  <Tool
    label="Cắt polygon"
    ariaLabel="polygon-split"
    shortcut={item.shortcut}
    active={item.selected}
    icon={<IconSplitTool />}
    tool={item}
    disabled={!item.splitTarget && !item.carveRegions}
    onClick={() => {
      // several polygons selected: carve the others out of the big one right away
      if (item.carveRegions) return item.carve();
      if (!item.selected) item.manager.selectTool(item, true);
    }}
  />
));

const _Tool = types
  .model("PolygonSplitTool", {
    group: "segmentation",
    shortcut: "X",
  })
  .volatile(() => ({
    canInteractWithRegions: false,
  }))
  .views((self) => ({
    get viewClass() {
      return () => <ToolView item={self} />;
    },

    // The selected polygon this tool may cut, or null
    get splitTarget() {
      const r = self.annotation?.highlightedNode;

      return r?.type === "polygonregion" &&
        r.closed &&
        !r.isReadOnly() &&
        r.parent?.name === self.obj?.name &&
        r.results.some((res) => res.from_name === self.control)
        ? r
        : null;
    },

    // Selected polygons to carve one out of another, or null
    get carveRegions() {
      return selectedPolygons(self);
    },
  }))
  .actions((self) => {
    // the area being drawn: target region, its contours in canvas coords, clicked points
    let cut = null;
    let layer = null;
    let line = null;
    let start = null;

    const tolerance = () => SNAP_PX / (self.obj?.zoomScale || 1);
    const near = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 <= tolerance() ** 2;

    return {
      // clicks on the outline and vertices must reach the tool instead of editing the polygon
      shouldSkipInteractions() {
        return true;
      },

      afterUpdateSelected() {
        if (!self.selected) self.cancel();
      },

      cancel() {
        cut = null;
        layer?.destroy();
        layer = line = start = null;
      },

      drawPreview(cursor) {
        const stage = self.obj?.stageRef;

        if (!cut || !stage) return;
        if (!layer) {
          const r = 4 / (self.obj.zoomScale || 1);

          layer = new Konva.Layer({ listening: false });
          line = new Konva.Line({ stroke: "#ff4d4f", strokeWidth: 2, dash: [6, 4], strokeScaleEnabled: false });
          // first point: click it again to close the area
          start = new Konva.Circle({ radius: r, fill: "#ff4d4f", x: cut.points[0][0], y: cut.points[0][1] });
          layer.add(line, start);
          stage.add(layer);
        }
        const pts = [...cut.points, ...(cursor ? [cursor] : [])];

        // an open cut line from a contour isn't closed back to its first point
        line.points((pts.length > 2 && !cut.start ? [...pts, cut.points[0]] : pts).flat());
        layer.batchDraw();
      },

      // selection changed under us (Escape, click elsewhere): drop the area
      targetLost() {
        if (cut && self.annotation.highlightedNode !== cut.region) self.cancel();
        return !cut;
      },

      mousemoveEv(_, [, , cx, cy]) {
        if (self.targetLost()) return;
        self.drawPreview([cx, cy]);
      },

      clickEv(_, [, , cx, cy]) {
        const pt = [cx, cy];

        if (self.targetLost()) {
          const region = self.splitTarget;

          if (!region) return;
          const polygon = regionGeometry(region);
          const rings = [polygon.points, polygon.holes].filter((r) => r.length);

          const start = snapToContour(rings, pt, tolerance());

          // start on a contour: the cut may also end on that contour (open line)
          cut = { region, polygon, rings, start, points: [start?.point ?? pt] };
          self.drawPreview(pt);
          return;
        }

        if (near(pt, cut.points[0])) {
          // back on the first point closes the drawn area
          if (cut.points.length > 2) self.finish();
          return;
        }

        const snap = snapToContour(cut.rings, pt, tolerance());

        if (cut.start && snap?.ring === cut.start.ring) return self.finish(snap);
        cut.points.push(snap?.point ?? pt);
        self.drawPreview(pt);
      },

      // end: contour snap ending an open cut line; without it the clicked points are a closed area
      finish(end) {
        const { region, polygon, rings, start, points } = cut;
        const res = end ? cutAlongLine(polygon, rings, start, end, points.slice(1)) : cutPolygon(polygon, points);

        self.cancel();
        if (res.error) {
          InfoModal.warning(res.error);
          return;
        }
        // new pieces first so the first created region is the cut-out one
        self.replaceRegion(region, [...res.inside, ...res.outside]);
      },

      carve() {
        // the hotkey fires even when the button is disabled
        const regions = self.carveRegions;

        if (!regions) return;
        const res = carvePolygons(regions.map(regionGeometry));

        if (res.error) {
          InfoModal.warning(res.error);
          return;
        }
        self.replaceRegion(regions[res.outerIndex], res.pieces);
      },

      // Replace a region by pieces given in canvas coords, keeping its results (labels) on each piece
      replaceRegion(region, canvasPieces) {
        // canvas coords -> percent, as PolygonRegion.serialize does
        const toPerc = (ring) => ring.map(([x, y]) => [region.convertXToPerc(x), region.convertYToPerc(y)]);
        const pieces = canvasPieces.map((p) => ({ points: toPerc(p.points), holes: toPerc(p.holes) }));
        const serialized = region.results.map((r) => r.serialize()).filter(Boolean);
        // shortcut: per-region results (textarea/choices) are copied to every piece unchanged
        const results = pieces.flatMap((piece, k) =>
          serialized.map((r) => ({ ...r, id: `${r.id}-${k}`, value: withGeometry(r.value, piece) })),
        );
        const { annotation } = self;

        annotation.history.freeze("polygon-split");
        const created = annotation.appendResults(results);

        // appendResults bails on read-only annotations; never delete the original then
        if (created?.length) region.deleteRegion();
        annotation.history.unfreeze("polygon-split");
        // select the new polygon so its label can be changed right away
        if (created?.length) annotation.selectAreas([created[0]]);
      },
    };
  });

const PolygonSplit = types.compose(_Tool.name, ToolMixin, BaseTool, _Tool);

export { PolygonSplit };
