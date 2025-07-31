import { isAlive, types } from "mobx-state-tree";

import BaseTool, { DEFAULT_DIMENSIONS } from "./Base";
import ToolMixin from "../mixins/Tool";
import { TwoPointsDrawingTool } from "../mixins/DrawingTool";
import { NodeViews } from "../components/Node/Node";
import { observe } from "mobx";
import { FF_DEV_2432, isFF } from "../utils/feature-flags";

const _Tool = types
  .model("EllipseBasedPolygonTool", {
    group: "segmentation",
    shortcut: "P",
  })
  .views((self) => {
    const Super = {
      createRegionOptions: self.createRegionOptions,
    };

    return {
      get getActivePolygon() {
        const poly = self.currentArea;

        if (isFF(FF_DEV_2432) && poly && !isAlive(poly)) return null;
        if (poly && poly.closed) return null;
        if (poly === undefined) return null;
        if (poly && poly.type !== "ellipsebasedpolygonregion") return null;

        return poly;
      },

      get tagTypes() {
        return {
          stateTypes: "ellipsebasedpolygonlabels",
          controlTagTypes: ["ellipsebasedpolygon", "ellipsebasedpolygonlabels"],
        };
      },

      get viewTooltip() {
        return "Ellipse Based Polygon region";
      },
      get iconComponent() {
        return self.dynamic ? NodeViews.EllipseBasedPolygonRegionModel.altIcon : NodeViews.EllipseBasedPolygonRegionModel.icon;
      },

      get defaultDimensions() {
        const { radius } = DEFAULT_DIMENSIONS.ellipse;

        return {
          width: radius,
          height: radius,
        };
      },

      createRegionOptions({ x, y }) {
        return Super.createRegionOptions({
          polygonPoints: [],
          x,
          y,
          radiusX: 1,
          radiusY: 1,
        });
      },
    };
  })
  .actions((self) => {
    const Super = {
      deleteRegion: self.deleteRegion,
    };
    return {
      commitDrawingRegion() {
        const { currentArea, control, obj } = self;

        if (!currentArea) return;
        const source = currentArea.toJSON();
        const value = Object.keys(currentArea.serialize().value).reduce(
          (value, key) => {
            value[key] = source[key];
            return value;
          },
          { coordstype: "px", dynamic: self.dynamic },
        );

        const [main, ...rest] = currentArea.results;
        const newArea = self.annotation.createResult(value, main.value.toJSON(), control, obj);

        //when user is using two different labels tag to draw a region, the other labels will be added to the region
        rest.forEach((r) => newArea.addResult(r.toJSON()));

        currentArea.setDrawing(false);
        self.deleteRegion();
        newArea.notifyDrawingFinished();
        return newArea;
      },
      beforeCommitDrawing() {
        const s = self.getActiveShape;

        return s.radiusX > self.MIN_SIZE.X && s.radiusY > self.MIN_SIZE.Y;
      },
    }
  });

const EllipseBasedPolygon = types.compose(_Tool.name, ToolMixin, BaseTool, TwoPointsDrawingTool, _Tool);

export { EllipseBasedPolygon };
