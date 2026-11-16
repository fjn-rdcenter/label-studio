import Konva from "konva";
import { observer } from "mobx-react";
import { types } from "mobx-state-tree";

import BaseTool from "./Base";
import ToolMixin from "../mixins/Tool";
import { Tool } from "../components/Toolbar/Tool";
import InfoModal from "../components/Infomodal/Infomodal";
import { IconSplitTool } from "../assets/icons";
import { pointInPolygon, snapToBoundary, splitPolygon } from "../utils/polygonOps";

// how close (screen px) a click must be to the outline to start or end the cut
const SNAP_PX = 8;

const ToolView = observer(({ item }) => (
  <Tool
    label="Cắt polygon"
    ariaLabel="polygon-split"
    shortcut={item.shortcut}
    active={item.selected}
    icon={<IconSplitTool />}
    tool={item}
    disabled={!item.splitTarget}
    onClick={() => {
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

      // shortcut: polygons with a hole (ring) can't be cut, extend splitPolygon to the hole if needed
      return r?.type === "polygonregion" &&
        r.closed &&
        !r.holes.length &&
        !r.isReadOnly() &&
        r.parent?.name === self.obj?.name &&
        r.results.some((res) => res.from_name === self.control)
        ? r
        : null;
    },
  }))
  .actions((self) => {
    // one cut in progress: target region, its outline in canvas coords, start snap, interior points
    let cut = null;
    let layer = null;
    let line = null;

    const tolerance = () => SNAP_PX / (self.obj?.zoomScale || 1);

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
        layer = null;
        line = null;
      },

      drawPreview(cursor) {
        const stage = self.obj?.stageRef;

        if (!cut || !stage) return;
        if (!layer) {
          layer = new Konva.Layer({ listening: false });
          line = new Konva.Line({ stroke: "#ff4d4f", strokeWidth: 2, dash: [6, 4], strokeScaleEnabled: false });
          layer.add(line);
          stage.add(layer);
        }
        line.points([cut.startPoint, ...cut.points, ...(cursor ? [cursor] : [])].flat());
        layer.batchDraw();
      },

      // selection changed under us (Escape, click elsewhere): drop the cut
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
          const ring = region.points.map((p) => [p.canvasX, p.canvasY]);
          const start = snapToBoundary(ring, pt, tolerance());

          if (!start) return;
          cut = { region, ring, start, startPoint: start.point ?? ring[start.index], points: [] };
          self.drawPreview(pt);
          return;
        }

        const end = snapToBoundary(cut.ring, pt, tolerance());

        if (end) return self.finish(end);
        if (pointInPolygon(pt, cut.ring)) {
          cut.points.push(pt);
          self.drawPreview(pt);
        }
      },

      finish(end) {
        const { region, ring, start, points } = cut;
        const res = splitPolygon(ring, start, end, points);

        self.cancel();
        if (res.error) {
          InfoModal.warning(res.error);
          return;
        }

        // canvas coords -> percent, as PolygonRegion.serialize does
        const toPerc = ([x, y]) => [region.convertXToPerc(x), region.convertYToPerc(y)];
        const serialized = region.results.map((r) => r.serialize()).filter(Boolean);
        // shortcut: per-region results (textarea/choices) are copied to both pieces unchanged
        const results = res.pieces.flatMap((piece, k) =>
          serialized.map((r) => ({ ...r, id: `${r.id}-${k}`, value: { ...r.value, points: piece.map(toPerc) } })),
        );
        const { annotation } = self;

        annotation.history.freeze("polygon-split");
        const created = annotation.appendResults(results);

        // appendResults bails on read-only annotations; never delete the original then
        if (created?.length) region.deleteRegion();
        annotation.history.unfreeze("polygon-split");
        // select the first piece so its label can be changed right away
        if (created?.length) annotation.selectAreas([created[0]]);
      },
    };
  });

const PolygonSplit = types.compose(_Tool.name, ToolMixin, BaseTool, _Tool);

export { PolygonSplit };
