import { types } from "mobx-state-tree";
import { observe } from "mobx";

import { IconRingPolygonTool } from "../assets/icons";
import { DrawingTool } from "../mixins/DrawingTool";
import ToolMixin from "../mixins/Tool";
import BaseTool, { DEFAULT_DIMENSIONS } from "./Base";

const RingPolygonBase = types
  .model("RingPolygonBase", {})
  .views((self) => ({
    get tagTypes() {
      return {
        stateTypes: "polygonlabels",
        controlTagTypes: ["polygonlabels", "polygon"],
      };
    },
    get defaultDimensions() {
      return DEFAULT_DIMENSIONS.polygon;
    },
    current() {
      return self.currentArea;
    },
    canStart() {
      return self.currentArea === null;
    },
    createRegionOptions({ x, y }) {
      return {
        points: [[x, y]],
        holes: [],
        ring: true,
        outerclosed: false,
        closed: false,
        dynamic: false,
      };
    },
  }))
  .actions((self) => {
    let lastPoint = { x: -1, y: -1 };
    let lastEvent = 0;
    let closedDisposer;
    const MOUSE_DOWN_EVENT = 1;
    const MOUSE_UP_EVENT = 2;
    const CLICK_EVENT = 3;

    const stopListening = () => {
      closedDisposer?.();
      closedDisposer = undefined;
    };

    const completeRing = () => {
      if (!self.isDrawing) return;
      stopListening();
      self.annotation.regionStore.selection.drawingUnselect();
      setTimeout(() => self._finishDrawing());
    };

    return {
      listenForClose() {
        stopListening();
        closedDisposer = observe(self.currentArea, "closed", () => {
          if (self.currentArea?.closed) completeRing();
        });
      },
      stopListening,
      handleToolSwitch() {
        stopListening();
        if (!self.isDrawing) return;
        self.deleteRegion();
        self._resetState();
      },
      mousemoveEv(_, [x, y]) {
        if (!self.isDrawing || !self.currentArea) return;
        const point = self.control?.getSnappedPoint({ x, y }) ?? { x, y };
        self.currentArea.setDrawingPoint(point.x, point.y);
      },
      mousedownEv(_, [x, y]) {
        lastPoint = { x, y };
        lastEvent = MOUSE_DOWN_EVENT;
      },
      mouseupEv(ev, [x, y]) {
        if (lastEvent === MOUSE_DOWN_EVENT && self.comparePointsWithThreshold(lastPoint, { x, y })) {
          self.placePoint(ev, x, y);
          lastEvent = MOUSE_UP_EVENT;
        }
        lastPoint = { x: -1, y: -1 };
      },
      clickEv(ev, [x, y]) {
        if (lastEvent !== MOUSE_UP_EVENT) self.placePoint(ev, x, y);
        lastEvent = CLICK_EVENT;
        lastPoint = { x: -1, y: -1 };
      },
      placePoint(_, x, y) {
        if (!self.currentArea) {
          if (!self.canStartDrawing() || self.annotation.regionStore.hasSelection) return;
          self.startDrawing(x, y);
          self.listenForClose();
          return;
        }

        const area = self.currentArea;
        if (area.outerclosed && area.holes.length === 0) {
          area.startInner(x, y);
          return;
        }

        const points = area.activePoints;
        if (points.length >= 3 && self.comparePointsWithThreshold(points[0], { x, y })) {
          area.closeActiveContour();
        } else {
          area.addRingPoint(x, y);
        }
      },
    };
  });

const RingPolygonIdentity = types
  .model("RingPolygonIdentity", {
    group: "segmentation",
    shortcut: "shift+p",
  })
  .views(() => ({
    get viewTooltip() {
      return "Ring Polygon (outer + inner)";
    },
    get iconComponent() {
      return IconRingPolygonTool;
    },
    get shouldRenderView() {
      return true;
    },
  }));

const RingPolygon = types.compose(
  "RingPolygonTool",
  ToolMixin,
  BaseTool,
  DrawingTool,
  RingPolygonBase,
  RingPolygonIdentity,
);

export { RingPolygon };
