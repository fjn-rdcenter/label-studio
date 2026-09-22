import { isAlive, types } from "mobx-state-tree";

import BaseTool, { DEFAULT_DIMENSIONS } from "./Base";
import ToolMixin from "../mixins/Tool";
import { MultipleClicksDrawingTool } from "../mixins/DrawingTool";
import { NodeViews } from "../components/Node/Node";
import { observe } from "mobx";
import { FF_DEV_2432, isFF } from "../utils/feature-flags";

const _Tool = types
  .model("PolygonToolBase", {})
  .views((self) => {
    const Super = {
      createRegionOptions: self.createRegionOptions,
      isIncorrectControl: self.isIncorrectControl,
      isIncorrectLabel: self.isIncorrectLabel,
    };

    return {
      get getActivePolygon() {
        const poly = self.currentArea;

        if (isFF(FF_DEV_2432) && poly && !isAlive(poly)) return null;
        if (poly && poly.closed) return null;
        if (poly === undefined) return null;
        if (poly && poly.type !== "polygonregion") return null;

        return poly;
      },

      get tagTypes() {
        return {
          stateTypes: "polygonlabels",
          controlTagTypes: ["polygonlabels", "polygon"],
        };
      },

      get defaultDimensions() {
        return DEFAULT_DIMENSIONS.polygon;
      },

      createRegionOptions({ x, y }) {
        return Super.createRegionOptions({
          points: [[x, y]],
          width: 10,
          closed: false,
          dynamic: self.dynamic || self.isOpenCV,
        });
      },

      isIncorrectControl() {
        return Super.isIncorrectControl() && self.current() === null;
      },
      isIncorrectLabel() {
        return !self.current() && Super.isIncorrectLabel();
      },
      canStart() {
        return self.current() === null;
      },

      current() {
        return self.getActivePolygon;
      },
    };
  })
  .actions((self) => {
    const Super = {
      startDrawing: self.startDrawing,
      _finishDrawing: self._finishDrawing,
      deleteRegion: self.deleteRegion,
    };

    let disposer;
    let closed;

    return {
      handleToolSwitch(tool) {
        self.stopListening();
        if (self.getCurrentArea()?.isDrawing && tool.toolName !== "ZoomPanTool") {
          const shape = self.getCurrentArea()?.toJSON();

          if (shape?.points?.length > 2) self.finishDrawing();
          else self.cleanupUncloseableShape();
        }
      },
      listenForClose() {
        closed = false;
        disposer = observe(
          self.getCurrentArea(),
          "closed",
          () => {
            if (self.getCurrentArea()?.closed && !closed) {
              self.finishDrawing();
            }
          },
          true,
        );
      },
      stopListening() {
        if (disposer) disposer();
      },
      closeCurrent() {
        self.stopListening();
        if (closed) return;
        closed = true;
        self.getCurrentArea().closePoly();
      },

      startDrawing(x, y) {
        const point = self.control?.getSnappedPoint({ x, y });

        if (isFF(FF_DEV_2432)) {
          self.mode = "drawing";
          self.currentArea = self.createRegion(self.createRegionOptions({ x: point.x, y: point.y }), true);
          self.setDrawing(true);
          self.applyActiveStates(self.currentArea);
        } else {
          Super.startDrawing(point.x, point.y);
        }
      },

      mousemoveEv(_, [x, y]) {
        const polygon = self.getCurrentArea();

        if (!self.isDrawing || !polygon || polygon.closed) return;

        const point = self.control?.getSnappedPoint({ x, y }) ?? { x, y };

        polygon.setDrawingPoint(point.x, point.y);
      },

      _finishDrawing() {
        if (isFF(FF_DEV_2432)) {
          const { currentArea, control } = self;

          self.setDrawing(false);
          self.currentArea.notifyDrawingFinished();
          self.currentArea = null;
          self.mode = "viewing";
          self.annotation.afterCreateResult(currentArea, control);
        } else {
          Super._finishDrawing();
        }
      },

      setDrawing(drawing) {
        self.currentArea?.setDrawing(drawing);
        self.annotation.setIsDrawing(drawing);
      },

      deleteRegion() {
        if (isFF(FF_DEV_2432)) {
          const { currentArea } = self;

          self.setDrawing(false);
          self.currentArea = null;
          if (currentArea) {
            currentArea.deleteRegion();
          }
        } else {
          Super.deleteRegion();
        }
      },
    };
  });

const PolygonIdentity = types
  .model("PolygonToolIdentity", {
    group: "segmentation",
    shortcut: "P",
  })
  .views(() => ({
    get viewTooltip() {
      return "Polygon region";
    },
    get iconComponent() {
      return NodeViews.PolygonRegionModel.icon;
    },
    get shouldRenderView() {
      return true;
    },
  }));

const OpenCVPolygonIdentity = types
  .model("OpenCVPolygonToolIdentity", {
    group: "opencv",
    shortcut: "I",
  })
  .volatile(() => ({
    isOpenCV: true,
  }))
  .views((self) => ({
    get viewTooltip() {
      return "OpenCV Detect Polygon";
    },
    get iconComponent() {
      return NodeViews.PolygonRegionModel.altIcon;
    },
    get shouldRenderView() {
      return self.control.smart;
    },
  }));

const Polygon = types.compose("PolygonTool", ToolMixin, BaseTool, MultipleClicksDrawingTool, _Tool, PolygonIdentity);
const OpenCVPolygon = types.compose(
  "OpenCVPolygonTool",
  ToolMixin,
  BaseTool,
  MultipleClicksDrawingTool,
  _Tool,
  OpenCVPolygonIdentity,
);

export { OpenCVPolygon, Polygon };
