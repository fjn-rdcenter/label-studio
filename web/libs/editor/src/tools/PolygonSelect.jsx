import Konva from "konva";
import { observer } from "mobx-react";
import { isAlive, types } from "mobx-state-tree";

import BaseTool from "./Base";
import ToolMixin from "../mixins/Tool";
import { Tool } from "../components/Toolbar/Tool";
import InfoModal from "../components/Infomodal/Infomodal";
import { IconSelectTool } from "../assets/icons";
import { clampDelta, pointsInRect } from "../utils/polygonOps";
import { editablePolygon } from "./PolygonMerge";

// how close (screen px) the pointer must be to a vertex to pick it
const SNAP_PX = 8;

const canvasXY = (p) => [p.canvasX, p.canvasY];

const ToolView = observer(({ item }) => (
  <Tool
    label="Chọn đỉnh (Select)"
    ariaLabel="polygon-select"
    shortcut={item.shortcut}
    active={item.selected}
    icon={<IconSelectTool />}
    tool={item}
    disabled={!item.target}
    onClick={() => {
      if (!item.selected) item.manager.selectTool(item, true);
    }}
  />
));

const _Tool = types
  .model("PolygonVertexSelectTool", {
    group: "segmentation",
    shortcut: "S",
  })
  .volatile(() => ({
    canInteractWithRegions: false,
  }))
  .views((self) => ({
    get viewClass() {
      return () => <ToolView item={self} />;
    },

    // The selected polygon whose vertices this tool edits, or null
    get target() {
      const r = self.annotation?.highlightedNode;

      return editablePolygon(self, r) ? r : null;
    },
  }))
  .actions((self) => {
    let region = null; // polygon whose vertices are selected
    let drag = null; // group move: last pointer position, vertex pressed, shift, whether anything moved
    let box = null; // box selection: start point, shift, vertex under the pointer
    let layer = null;
    let rect = null;

    const tolerance = () => SNAP_PX / (self.obj?.zoomScale || 1);
    const vertices = () => (region && isAlive(region) ? [...region.points, ...region.holes] : []);
    const selectedVertices = () => vertices().filter((p) => p.selected);

    const nearestVertex = (pt) => {
      let best = null;

      vertices().forEach((p) => {
        const d = Math.hypot(p.canvasX - pt[0], p.canvasY - pt[1]);

        if (d <= tolerance() && (!best || d < best.d)) best = { p, d };
      });
      return best?.p ?? null;
    };

    // Backspace/Delete removes the selected vertices instead of the whole region
    const onKeyDown = (e) => {
      if (e.key !== "Backspace" && e.key !== "Delete") return;
      if (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target?.tagName) || e.target?.isContentEditable) return;
      self.syncTarget();
      if (!selectedVertices().length) return;
      e.preventDefault();
      e.stopImmediatePropagation();
      self.deleteSelected();
    };

    return {
      // clicks and drags on vertices and edges must reach the tool instead of editing the polygon
      shouldSkipInteractions() {
        return true;
      },

      afterUpdateSelected() {
        if (self.selected) {
          window.addEventListener("keydown", onKeyDown, true);
          // start fresh: vertices picked with the Move tool must not join the selection
          region = null;
          self.syncTarget();
        } else {
          window.removeEventListener("keydown", onKeyDown, true);
          self.endDrag();
          self.clearSelection();
          self.removeBox();
          region = null;
        }
      },

      beforeDestroy() {
        window.removeEventListener("keydown", onKeyDown, true);
        self.endDrag();
      },

      // release the history lock taken on mousedown, even if mouseup never comes
      endDrag() {
        if (!drag) return;
        const { moved } = drag;

        drag = null;
        self.annotation?.history.unfreeze("polygon-select");
        if (moved && region && isAlive(region)) region.notifyPointsChanged();
      },

      clearSelection() {
        vertices().forEach((p) => p.selected && p.setSelected(false));
      },

      // a new polygon got selected (or none): drop the vertex selection of the previous one
      syncTarget() {
        if (region === self.target) return;
        self.clearSelection();
        region = self.target;
        // PolygonPoint.selected is shared with the Move tool's selected point: start with none
        self.clearSelection();
      },

      removeBox() {
        layer?.destroy();
        layer = rect = box = null;
      },

      drawBox(a, b) {
        const stage = self.obj?.stageRef;

        if (!stage) return;
        if (!layer) {
          layer = new Konva.Layer({ listening: false });
          rect = new Konva.Rect({ stroke: "#1890ff", strokeWidth: 1, dash: [4, 3], strokeScaleEnabled: false });
          layer.add(rect);
          stage.add(layer);
        }
        rect.setAttrs({
          x: Math.min(a[0], b[0]),
          y: Math.min(a[1], b[1]),
          width: Math.abs(b[0] - a[0]),
          height: Math.abs(b[1] - a[1]),
        });
        layer.batchDraw();
      },

      mousedownEv(ev, [, , cx, cy]) {
        self.syncTarget();
        if (!region) return;
        const pt = [cx, cy];
        const hit = nearestVertex(pt);

        if (hit?.selected) {
          drag = { last: pt, hit, shift: ev.shiftKey, moved: false };
          self.annotation.history.freeze("polygon-select");
          return;
        }
        box = { start: pt, shift: ev.shiftKey, hit };
      },

      mousemoveEv(_, [, , cx, cy]) {
        if (drag) {
          const moving = selectedVertices();
          const [dx, dy] = clampDelta(
            moving.map(canvasXY),
            cx - drag.last[0],
            cy - drag.last[1],
            self.obj.stageWidth,
            self.obj.stageHeight,
          );

          if (dx || dy) drag.moved = true;
          moving.forEach((p) => p.movePoint(dx, dy));
          drag.last = [drag.last[0] + dx, drag.last[1] + dy];
          return;
        }
        if (box) self.drawBox(box.start, [cx, cy]);
      },

      mouseupEv(_, [, , cx, cy]) {
        if (drag) {
          const { moved, hit, shift } = drag;

          self.endDrag();
          if (moved) return;
          // a click on a selected vertex: Shift deselects it, otherwise it becomes the only one
          if (shift) {
            hit.setSelected(false);
          } else {
            self.clearSelection();
            hit.setSelected(true);
          }
          return;
        }
        if (!box) return;
        const { start, shift, hit } = box;
        const pt = [cx, cy];

        self.removeBox();
        if (Math.hypot(pt[0] - start[0], pt[1] - start[1]) < 3 / (self.obj?.zoomScale || 1)) {
          // a click: pick the vertex under the pointer (Shift toggles), or clear on empty space
          if (!shift) self.clearSelection();
          if (hit) hit.setSelected(shift ? !hit.selected : true);
          return;
        }
        if (!shift) self.clearSelection();
        const all = vertices();

        pointsInRect(all.map(canvasXY), start, pt).forEach((i) => all[i].setSelected(true));
      },

      deleteSelected() {
        const doomed = selectedVertices();
        const { history } = self.annotation;

        history.freeze("polygon-select");
        // deletePoint keeps at least 3 vertices per contour; the others stay selected
        doomed.forEach((p) => region.deletePoint(p));
        history.unfreeze("polygon-select");

        const kept = doomed.filter(isAlive).length;

        if (kept) InfoModal.warning(`Mỗi đường viền cần ít nhất 3 đỉnh, ${kept} đỉnh được giữ lại.`);
      },
    };
  });

const PolygonSelect = types.compose(_Tool.name, ToolMixin, BaseTool, _Tool);

export { PolygonSelect };
