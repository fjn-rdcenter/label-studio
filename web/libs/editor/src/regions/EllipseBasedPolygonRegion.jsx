import React, { Fragment, memo, useContext, useEffect, useMemo } from "react";
import { Ellipse, Group, Label, Line } from "react-konva";
import { destroy, detach, getRoot, isAlive, types } from "mobx-state-tree";

import Constants from "../core/Constants";
import Registry from "../core/Registry";
import NormalizationMixin from "../mixins/Normalization";
import RegionsMixin from "../mixins/Regions";

import { ImageViewContext } from "../components/ImageView/ImageViewContext";
import { LabelOnEllipse } from "../components/ImageView/LabelOnRegion";
import { guidGenerator } from "../core/Helpers";
import { useRegionStyles } from "../hooks/useRegionColor";
import { AreaMixin } from "../mixins/AreaMixin";
import { KonvaRegionMixin } from "../mixins/KonvaRegion";
import { ImageModel } from "../tags/object/Image";
import { rotateBboxCoords } from "../utils/bboxCoords";
import { FF_DEV_3793, FF_DEV_2432, isFF } from "../utils/feature-flags";
import { createDragBoundFunc } from "../utils/image";
import { AliveRegion } from "./AliveRegion";
import { EditableRegion } from "./EditableRegion";
import {
  RELATIVE_STAGE_HEIGHT,
  RELATIVE_STAGE_WIDTH
} from "../components/ImageView/Image";

import { PolygonPoint, PolygonPointView } from "./PolygonPoint";
import { observer } from "mobx-react";
import { LabelOnPolygon } from "../components/ImageView/LabelOnRegion";
import { green } from "@ant-design/colors"; 


// This is a mixed region from Ellipse and PolygonRegion
// It is used to create a polygon region based on an ellipse shape
// After drawing ellipse will disappear and polygon will be shown
// Transform of ellipse is being disabled, view useTransformer, not used rotation and scale, if used need to change calculation for generate polygon points
// Polygon points are editable, but just draggable. Feature like add new point in line, dbl click to remove point are disabled for now, you can view at deletePoint.
// Important: detector used polygonPoint, radiusX, radiusY  to detect this region, so you need to define those properties in tools (view tools/EllipseBasedPolygon.js)
// Detection process at Registry.ts/getAvailableAreas

const ellipsePerimeterRamanujan = (a, b) => {
  const pi = Math.PI;
  return pi * (3 * (a + b) - Math.sqrt((3 * a + b) * (a + 3 * b)));
};

const generatePointsFromEllipse = (canvasX, canvasY, canvasRadiusX, canvasRadiusY, stageWidth, stageHeight, coordstype) => {
  const points = [];
  const gapPoint = 25; // Distance between points
  const circumference = ellipsePerimeterRamanujan(canvasRadiusX, canvasRadiusY);
  let numberOfPoints = Math.ceil(circumference / gapPoint);
  if (numberOfPoints < 12){
    numberOfPoints = 12; // Minimum number of points to create a smooth ellipse
  }
  const step = 360 / numberOfPoints;
  for (let angle = 0; angle < 360; angle += step) {
    const rad = (angle * Math.PI) / 180;
    const xCoord = canvasX + canvasRadiusX * Math.cos(rad);
    const yCoord = canvasY + canvasRadiusY * Math.sin(rad);
    points.push({ x: xCoord, y: yCoord });
  }
  if(coordstype === "px"){
    return points;
  }
  return points.map(point => ({
    x: (point.x / stageWidth) * RELATIVE_STAGE_WIDTH,
    y: (point.y / stageHeight) * RELATIVE_STAGE_HEIGHT
  }));
}

const EllipseBasedPolygonRegionAbsoluteCoordsDEV3793 = types
  .model({
    coordstype: types.optional(types.enumeration(["px", "perc"]), "perc")
  })
  .volatile(() => ({
    relativeX: 0,
    relativeY: 0,
    relativeWidth: 0,
    relativeHeight: 0,
    relativeRadiusX: 0,
    relativeRadiusY: 0,
    pointSize: 'small',
    pointStyle: 'circle',
  }))
  .actions(self => ({
    afterCreate() {
      self.startX = self.x;
      self.startY = self.y;

      switch (self.coordstype) {
        case "perc": {
          self.relativeX = self.x;
          self.relativeY = self.y;
          self.relativeRadiusX = self.radiusX;
          self.relativeRadiusY = self.radiusY;
          self.relativeWidth = self.width;
          self.relativeHeight = self.height;
          break;
        }
        case "px": {
          const { stageWidth, stageHeight } = self.parent;

          if (stageWidth && stageHeight) {
            self.setPosition(
              self.x,
              self.y,
              self.radiusX,
              self.radiusY,
              self.rotation
            );
          }
          break;
        }
      }
      self.checkSizes();
      self.updateAppearenceFromState();
    },
    setPosition(x, y, radiusX, radiusY, rotation) {
      self.x = x;
      self.y = y;
      self.radiusX = radiusX;
      self.radiusY = radiusY;

      self.relativeX = (x / self.parent?.stageWidth) * RELATIVE_STAGE_WIDTH;
      self.relativeY = (y / self.parent?.stageHeight) * RELATIVE_STAGE_HEIGHT;

      self.relativeRadiusX =
        (radiusX / self.parent?.stageWidth) * RELATIVE_STAGE_WIDTH;
      self.relativeRadiusY =
        (radiusY / self.parent?.stageHeight) * RELATIVE_STAGE_HEIGHT;

      self.rotation = (rotation + 360) % 360;
      self.generatePolygonPointsFromEllipse()
    },
    setPositionInternal(x, y, radiusX, radiusY, rotation) {
      return self.setPosition(x, y, radiusX, radiusY, rotation);
    },
    generatePolygonPointsFromEllipse() {
      const points = generatePointsFromEllipse(self.canvasX, self.canvasY, self.canvasRadiusX, self.canvasRadiusY, self.parent?.stageWidth, self.parent?.stageHeight, self.coordstype);
      self.polygonPoints = points.map((p, index) => ({
        id: guidGenerator(),
        x: p.x,
        y: p.y,
        size: self.pointSize,
        style: self.pointStyle,
        index
      }));
      return self.polygonPoints;
    },
    updateImageSize(wp, hp, sw, sh) {

      // update for ellipse region
      self.sw = sw;
      self.sh = sh;

      if (self.coordstype === "px") {
        self.x = (sw * self.relativeX) / RELATIVE_STAGE_WIDTH;
        self.y = (sh * self.relativeY) / RELATIVE_STAGE_HEIGHT;
        self.radiusX = (sw * self.relativeRadiusX) / RELATIVE_STAGE_WIDTH;
        self.radiusY = (sh * self.relativeRadiusY) / RELATIVE_STAGE_HEIGHT;
      } else if (self.coordstype === "perc") {
        self.x = (sw * self.x) / RELATIVE_STAGE_WIDTH;
        self.y = (sh * self.y) / RELATIVE_STAGE_HEIGHT;
        self.radiusX = (sw * self.radiusX) / RELATIVE_STAGE_WIDTH;
        self.radiusY = (sh * self.radiusY) / RELATIVE_STAGE_HEIGHT;
        self.coordstype = "px";
      }
      // update polygon points
      if (self.coordstype === "px") {
        self.polygonPoints.forEach((p) => {
          const x = (sw * p.relativeX) / RELATIVE_STAGE_WIDTH;
          const y = (sh * p.relativeY) / RELATIVE_STAGE_HEIGHT;

          p._setPos(x, y);
        });
      }

      if (!self.annotation.sentUserGenerate && self.coordstype === "perc") {
        self.polygonPoints.forEach((p) => {
          const x = (sw * p.x) / RELATIVE_STAGE_WIDTH;
          const y = (sh * p.y) / RELATIVE_STAGE_HEIGHT;

          self.coordstype = "px";
          p._setPos(x, y);
        });
      }

    }
  }));

/**
 * Ellipse object for Bounding Box
 *
 */
const Model = types
  .model({
    id: types.optional(types.identifier, guidGenerator),
    pid: types.optional(types.string, guidGenerator),
    type: "ellipsebasedpolygonregion",
    object: types.late(() => types.reference(ImageModel)),

    x: types.number,
    y: types.number,
    radiusX: types.number,
    radiusY: types.number,

    rotation: 0,

    polygonPoints: types.array(
      types.union(PolygonPoint, types.array(types.number)),
      []
    ),
    width: 10,
    closed: true
  })
  .volatile(() => ({
    startX: 0,
    startY: 0,

    // @todo not used
    scaleX: 1,
    scaleY: 1,

    opacity: types.number,

    fill: true,
    fillColor: Constants.FILL_COLOR,
    fillOpacity: 0.2,

    strokeColor: Constants.STROKE_COLOR,
    strokeWidth: Constants.STROKE_WIDTH,

    _supportsTransform: true,
    hideable: true,
    mouseOverStartPoint: false,

  }))
  .volatile(() => {
    return {
      useTransformer: false, // whether to use transformer for this region, mean display bbox and used this for resizing region
      preferTransformer: false,
      supportsRotate: false,
      supportsScale: false,
      selectedPolygonPoint: null,
      pointSize: 'small',
      pointStyle: 'circle',
    };
  })
  .views(self => ({
    get store() {
      return getRoot(self);
    },
    get bboxCoords() {
      const bboxCoords = {
        left: self.x - self.radiusX,
        top: self.y - self.radiusY,
        right: self.x + self.radiusX,
        bottom: self.y + self.radiusY
      };

      if (self.rotation === 0) return bboxCoords;

      return rotateBboxCoords(
        bboxCoords,
        self.rotation,
        { x: self.x, y: self.y },
        self.parent.whRatio
      );
    },
    get canvasX() {
      return isFF(FF_DEV_3793)
        ? self.parent?.internalToCanvasX(self.x)
        : self.x;
    },
    get canvasY() {
      return isFF(FF_DEV_3793)
        ? self.parent?.internalToCanvasY(self.y)
        : self.y;
    },
    get canvasRadiusX() {
      return isFF(FF_DEV_3793)
        ? self.parent?.internalToCanvasX(self.radiusX)
        : self.radiusX;
    },
    get canvasRadiusY() {
      return isFF(FF_DEV_3793)
        ? self.parent?.internalToCanvasY(self.radiusY)
        : self.radiusY;
    },
    get flattenedPoints() {
      return getFlattenedPoints(self.polygonPoints);
    }
  }))
  .actions(self => ({
    afterCreate() {
      self.startX = self.x;
      self.startY = self.y;
      if (!self.polygonPoints.length) return;
      if (!self.polygonPoints[0].id) {
        self.polygonPoints = self.polygonPoints.map(([x, y], index) => ({
          id: guidGenerator(),
          x,
          y,
          size: self.pointSize,
          style: self.pointStyle,
          index,
        }));
      }
      if (!isFF(FF_DEV_2432)) self.closed = self.polygonPoints.length > 2;
      self.checkSizes();
    },

    // @todo not used
    coordsInside(x, y) {
      // check if x and y are inside the rectangle
      const a = self.radiusX;
      const b = self.radiusY;

      const cx = self.x;
      const cy = self.y;
      //going to system where center coordinates are (0,0)
      let rel_x = x - cx;
      let rel_y = y - cy;

      //going to system where our ellipse has angle 0 to X-Axis via rotate matrix
      const theta = self.rotation;

      rel_x =
        rel_x * Math.cos(Math.unit(theta, "deg")) -
        rel_y * Math.sin(Math.unit(theta, "deg"));
      rel_y =
        rel_x * Math.sin(Math.unit(theta, "deg")) +
        rel_y * Math.cos(Math.unit(theta, "deg"));

      if (Math.abs(rel_x) < a) {
        if (rel_y ** 2 < b ** 2 * (1 - rel_x ** 2 / a ** 2)) {
          return true;
        }
      } else {
        return false;
      }
    },

    setPositionInternal(x, y, radiusX, radiusY, rotation) {
      self.x = x;
      self.y = y;
      self.radiusX = radiusX;
      self.radiusY = radiusY;
      self.rotation = (rotation + 360) % 360;
      self.generatePolygonPointsFromEllipse();
    },
    generatePolygonPointsFromEllipse() {
      const points = generatePointsFromEllipse(self.canvasX, self.canvasY, self.canvasRadiusX, self.canvasRadiusY, self.parent?.stageWidth, self.parent?.stageHeight, self.coordstype);
      self.polygonPoints = points.map((p, index) => ({
        id: guidGenerator(),
        x: p.x,
        y: p.y,
        size: self.pointSize,
        style: self.pointStyle,
        index
      }));
      return self.polygonPoints;
    },

    /**
     * Boundg Box set position on canvas
     * @param {number} x
     * @param {number} y
     * @param {number} radiusX
     * @param {number} radiusY
     * @param {number} rotation
     */
    setPosition(x, y, radiusX, radiusY, rotation) {
      self.setPositionInternal(
        self.parent.canvasToInternalX(x),
        self.parent.canvasToInternalY(y),
        self.parent.canvasToInternalX(radiusX),
        self.parent.canvasToInternalY(radiusY),
        rotation
      );
    },

    setScale(x, y) {
      self.scaleX = x;
      self.scaleY = y;
    },

    setFill(color) {
      self.fill = color;
    },

    updateImageSize() {},

    //========================================
    // Actions for polygon regions, we need to update the polygon points
    //======================================

    setMouseOverStartPoint(value) {
      self.mouseOverStartPoint = value;
    },
    handleMouseMove({ e, flattenedPoints }) {
      const { offsetX, offsetY } = e.evt;
      const [cursorX, cursorY] = self.parent.fixZoomedCoords([
        offsetX,
        offsetY
      ]);
      const [x, y] = getAnchorPoint({ flattenedPoints, cursorX, cursorY });

      const group = e.currentTarget;
      const layer = e.currentTarget.getLayer();
      const zoom = self.parent.zoomScale;

      moveHoverAnchor({ point: [x, y], group, layer, zoom });
    },

    handleMouseLeave({ e }) {
      removeHoverAnchor({ layer: e.currentTarget.getLayer() });
    },

    insertPoint(insertIdx, x, y) {
      const pointCoords = self.control?.getSnappedPoint({
        x: self.parent.canvasToInternalX(x),
        y: self.parent.canvasToInternalY(y)
      });
      const isMatchWithPrevPoint =
        self.polygonPoints[insertIdx - 1] &&
        self.parent.isSamePixel(pointCoords, self.polygonPoints[insertIdx - 1]);
      const isMatchWithNextPoint =
        self.polygonPoints[insertIdx] &&
        self.parent.isSamePixel(pointCoords, self.polygonPoints[insertIdx]);

      if (isMatchWithPrevPoint || isMatchWithNextPoint) {
        return;
      }

      const p = {
        id: guidGenerator(),
        x: pointCoords.x,
        y: pointCoords.y,
        size: self.pointSize,
        style: self.pointStyle,
        index: self.polygonPoints.length
      };

      self.polygonPoints.splice(insertIdx, 0, p);

      return self.polygonPoints[insertIdx];
    },

    setPolygonPoints(points) {
      self.polygonPoints.forEach((p, idx) => {
        p.x = points[idx * 2];
        p.y = points[idx * 2 + 1];
      });
    },

    handleLineClick({ e, flattenedPoints, insertIdx }) {
      if (!self.closed || !self.selected) return;

      e.cancelBubble = true;

      removeHoverAnchor({ layer: e.currentTarget.getLayer() });

      const { offsetX, offsetY } = e.evt;

      const [cursorX, cursorY] = self.parent.fixZoomedCoords([
        offsetX,
        offsetY
      ]);
      const point = getAnchorPoint({ flattenedPoints, cursorX, cursorY });

      self.insertPoint(insertIdx, point[0], point[1]);
    },

    setSelectedPoint(point) {
        if (self.selectedPolygonPoint) {
          self.selectedPolygonPoint.selected = false;
        }

        point.selected = true;
        self.selectedPolygonPoint = point;
    },

    
    deletePoint(point) {
      return; // disable point deletion for now
      const willNotEliminateClosedShape = self.polygonPoints.length <= 3 && point.parent.closed;
      const isLastPoint = self.polygonPoints.length === 1;
      const isSelected = self.selectedPolygonPoint === point;

      if (willNotEliminateClosedShape || isLastPoint) return;
      if (isSelected) self.selectedPolygonPoint = null;
      destroy(point);
    },

    /**
     * @example
     * {
     *   "original_width": 1920,
     *   "original_height": 1280,
     *   "image_rotation": 0,
     *   "value": {
     *     "polygonPoints": [
     *       [3.1, 8.2],
     *       [4.5, 9.0]
     *     ],
     *     "ellipsebasedpolygonlabels": ["Car"]
     *   }
     * }
     * @typedef {Object} EllipseBasedPolygonRegionResult
     * @property {number} original_width width of the original image (px)
     * @property {number} original_height height of the original image (px)
     * @property {number} image_rotation rotation degree of the image (deg)
     * @property {Object} value
     * @property {number[][]} value.points list of (x, y) coordinates of the polygon by percentage of the image size (0-100)
     */

    /**
     * @return {EllipseBasedPolygonRegionResult}
     */
    serialize() {
      const value = {
        x: isFF(FF_DEV_3793) ? self.x : self.convertXToPerc(self.x),
        y: isFF(FF_DEV_3793) ? self.y : self.convertYToPerc(self.y),
        radiusX: isFF(FF_DEV_3793)
          ? self.radiusX
          : self.convertHDimensionToPerc(self.radiusX),
        radiusY: isFF(FF_DEV_3793)
          ? self.radiusY
          : self.convertVDimensionToPerc(self.radiusY),
        rotation: self.rotation,
        polygonPoints: isFF(FF_DEV_3793)
          ? self.polygonPoints.map(p => [p.x, p.y])
          : self.polygonPoints.map(p => [
              self.convertXToPerc(p.x),
              self.convertYToPerc(p.y)
            ]),
        ...(isFF(FF_DEV_2432) ? { closed: self.closed } : {})
      };

      return self.parent.createSerializedResult(self, value);
    },

     /**
     * @example
     * {
     *   "original_width": 1920,
     *   "original_height": 1280,
     *   "image_rotation": 0,
     *   "value": {
     *     "x": 3.1,
     *     "y": 8.2,
     *     "radiusX": 20,
     *     "radiusY": 16,
     *     "polygonPoints": [
     *       [3.1, 8.2],
     *       [4.5, 9.0]
     *     ],
     *     "ellipsebasedpolygonlabels": ["Car"]
     *   }
     * }
     * @typedef {Object} EllipseBasedPolygonRegionResultForDrawing
     * @property {number} original_width width of the original image (px)
     * @property {number} original_height height of the original image (px)
     * @property {number} image_rotation rotation degree of the image (deg)
     * @property {Object} value
     * @property {number} value.x x coordinate of the top left corner before rotation (0-100)
     * @property {number} value.y y coordinate of the top left corner before rotation (0-100)
     * @property {number} value.radiusX radius by x axis (0-100)
     * @property {number} value.radiusY radius by y axis (0-100)
     * @property {number} value.rotation rotation degree (deg)
     * @property {number[][]} value.points list of (x, y) coordinates of the polygon by percentage of the image size (0-100)
     */

    /**
     * @return {EllipseBasedPolygonRegionResultForDrawing}
     */
    serializeForDrawing() {
      const value = {
        x: isFF(FF_DEV_3793) ? self.x : self.convertXToPerc(self.x),
        y: isFF(FF_DEV_3793) ? self.y : self.convertYToPerc(self.y),
        radiusX: isFF(FF_DEV_3793)
          ? self.radiusX
          : self.convertHDimensionToPerc(self.radiusX),
        radiusY: isFF(FF_DEV_3793)
          ? self.radiusY
          : self.convertVDimensionToPerc(self.radiusY),
        rotation: self.rotation,
      };

      return self.parent.createSerializedResult(self, value);
    }
  }));

const EllipseBasedPolygonRegionModel = types.compose(
  "EllipseBasedPolygonRegionModel",
  RegionsMixin,
  AreaMixin,
  NormalizationMixin,
  KonvaRegionMixin,
  EditableRegion,
  Model,
  ...(isFF(FF_DEV_3793) ? [] : [EllipseBasedPolygonRegionAbsoluteCoordsDEV3793])
);

/**
 * Get coordinates of anchor point
 * @param {array} flattenedPoints
 * @param {number} cursorX coordinates of cursor X
 * @param {number} cursorY coordinates of cursor Y
 */
function getAnchorPoint({ flattenedPoints, cursorX, cursorY }) {
  const [point1X, point1Y, point2X, point2Y] = flattenedPoints;
  const y =
    ((point2X - point1X) * (point2X * point1Y - point1X * point2Y) +
      (point2X - point1X) * (point2Y - point1Y) * cursorX +
      (point2Y - point1Y) * (point2Y - point1Y) * cursorY) /
    ((point2Y - point1Y) * (point2Y - point1Y) +
      (point2X - point1X) * (point2X - point1X));
  const x =
    cursorX -
    ((point2Y - point1Y) *
      (point2X * point1Y -
        point1X * point2Y +
        cursorX * (point2Y - point1Y) -
        cursorY * (point2X - point1X))) /
      ((point2Y - point1Y) * (point2Y - point1Y) +
        (point2X - point1X) * (point2X - point1X));

  return [x, y];
}

function getFlattenedPoints(points) {
  const p = points.map(p => [p.canvasX, p.canvasY]);

  return p.reduce(
    (flattenedPoints, point) => flattenedPoints.concat(point),
    []
  );
}

function getHoverAnchor({ layer }) {
  return layer.findOne(".hoverAnchor");
}

/**
 * Create new anchor for current polygon
 */
function createHoverAnchor({ point, group, layer, zoom }) {
  const hoverAnchor = new Konva.Circle({
    name: "hoverAnchor",
    x: point[0],
    y: point[1],
    stroke: green.primary,
    fill: green[0],
    scaleX: 1 / (zoom || 1),
    scaleY: 1 / (zoom || 1),

    strokeWidth: 2,
    radius: 5
  });

  group.add(hoverAnchor);
  layer.draw();
  return hoverAnchor;
}

function moveHoverAnchor({ point, group, layer, zoom }) {
  const hoverAnchor =
    getHoverAnchor({ layer }) ||
    createHoverAnchor({ point, group, layer, zoom });

  hoverAnchor.to({ x: point[0], y: point[1], duration: 0 });
}

function removeHoverAnchor({ layer }) {
  const hoverAnchor = getHoverAnchor({ layer });

  if (!hoverAnchor) return;
  hoverAnchor.destroy();
  layer.draw();
}

const Poly = memo(
  observer(({ item, colors, dragProps, draggable }) => {
    const { flattenedPoints } = item;
    const name = "poly";

    return (
      <Group key={name} name={name}>
        <Line
          name="_transformable"
          lineJoin="round"
          lineCap="square"
          stroke={colors.strokeColor}
          strokeWidth={colors.strokeWidth}
          strokeScaleEnabled={false}
          perfectDrawEnabled={false}
          shadowForStrokeEnabled={false}
          points={flattenedPoints}
          fill={colors.fillColor}
          closed={true}
          {...dragProps}
          onTransformEnd={e => {
            if (e.target !== e.currentTarget) return;

            const t = e.target;

            const d = [t.getAttr("x", 0), t.getAttr("y", 0)];
            const scale = [t.getAttr("scaleX", 1), t.getAttr("scaleY", 1)];
            const points = t.getAttr("points");

            item.setPolygonPoints(
              points.reduce((result, coord, idx) => {
                const isXCoord = idx % 2 === 0;

                if (isXCoord) {
                  const point = item.control?.getSnappedPoint({
                    x: item.parent.canvasToInternalX(coord * scale[0] + d[0]),
                    y: item.parent.canvasToInternalY(
                      points[idx + 1] * scale[1] + d[1]
                    )
                  });

                  result.push(point.x, point.y);
                }
                return result;
              }, [])
            );

            t.setAttr("x", 0);
            t.setAttr("y", 0);
            t.setAttr("scaleX", 1);
            t.setAttr("scaleY", 1);
          }}
          draggable={draggable}
        />
      </Group>
    );
  })
);

/**
 * Line between 2 points
 */
const Edge = observer(({ name, item, idx, p1, p2, closed, regionStyles }) => {
  const insertIdx = idx + 1; // idx1 + 1 or idx2
  const flattenedPoints = [p1.canvasX, p1.canvasY, p2.canvasX, p2.canvasY];

  const lineProps = closed
    ? {
        stroke: "transparent",
        strokeWidth: regionStyles.strokeWidth,
        strokeScaleEnabled: false
      }
    : {
        stroke: regionStyles.strokeColor,
        strokeWidth: regionStyles.strokeWidth,
        strokeScaleEnabled: false
      };

  return (
    <Group
      key={name}
      name={name}
      // onClick={e => item.handleLineClick({ e, flattenedPoints, insertIdx })} // disabled line interaction for now
      // onMouseMove={e => {
      //   if (!item.closed || !item.selected || item.isReadOnly()) return;

      //   item.handleMouseMove({ e, flattenedPoints });
      // }}
      // onMouseLeave={e => item.handleMouseLeave({ e })}
    >
      <Line
        lineJoin="round"
        opacity={1}
        points={flattenedPoints}
        hitStrokeWidth={20}
        strokeScaleEnabled={false}
        perfectDrawEnabled={false}
        shadowForStrokeEnabled={false}
        {...lineProps}
      />
    </Group>
  );
});

const Edges = memo(
  observer(({ item, regionStyles }) => {
    const { polygonPoints, closed } = item;
    const name = "borders";

    if (item.closed && (item.parent.useTransformer || !item.selected)) {
      return null;
    }
    return (
      <Group key={name} name={name}>
        {polygonPoints.map((p, idx) => {
          const idx1 = idx;
          const idx2 = idx === polygonPoints.length - 1 ? 0 : idx + 1;

          if (!closed && idx2 === 0) {
            return null;
          }

          return (
            <Edge
              key={`border_${idx1}_${idx2}`}
              name={`border_${idx1}_${idx2}`}
              item={item}
              idx={idx1}
              p1={polygonPoints[idx]}
              p2={polygonPoints[idx2]}
              closed={closed}
              regionStyles={regionStyles}
            />
          );
        })}
      </Group>
    );
  })
);

const HtxEllipseBasedPolygonView = ({ item, setShapeRef }) => {
  const { store } = item;

  const regionStyles = useRegionStyles(item, {
    useStrokeAsFill: true
  });
  const stage = item.parent?.stageRef;
  const { suggestion } = useContext(ImageViewContext) ?? {};

  function renderCircle({ points, idx }) {
    const name = `anchor_${points.length}_${idx}`;
    const point = points[idx];

    if (!item.closed || (item.closed && item.selected)) {
      return <PolygonPointView item={point} name={name} key={name} />;
    }
  }

  function renderCircles(points) {
    const name = "anchors";

    if (item.closed && (item.parent.useTransformer || !item.selected)) {
      return null;
    }
    return (
      <Group key={name} name={name}>
        {points.map((p, idx) => renderCircle({ points, idx }))}
      </Group>
    );
  }

  const dragProps = useMemo(() => {
    let isDragging = false;

    return {
      onDragStart: e => {
        if (e.target !== e.currentTarget) return;
        if (item.parent.getSkipInteractions()) {
          e.currentTarget.stopDrag(e.evt);
          return;
        }
        isDragging = true;
        item.annotation.setDragMode(true);

        item.annotation.history.freeze(item.id);
      },
      dragBoundFunc: createDragBoundFunc(item, {
        x: -item.bboxCoords.left,
        y: -item.bboxCoords.top
      }),
      onDragEnd: e => {
        if (!isDragging) return;
        const t = e.target;

        if (e.target === e.currentTarget) {
          item.annotation.setDragMode(false);

          const point = item.control?.getSnappedPoint({
            x: item.parent?.canvasToInternalX(t.getAttr("x")),
            y: item.parent?.canvasToInternalY(t.getAttr("y"))
          });

          point.x = item.parent?.internalToCanvasX(point.x);
          point.y = item.parent?.internalToCanvasY(point.y);

          item.polygonPoints.forEach(p => p.movePoint(point.x, point.y));
          item.annotation.history.unfreeze(item.id);
        }

        t.setAttr("x", 0);
        t.setAttr("y", 0);
        isDragging = false;
      }
    };
  }, [item.bboxCoords.left, item.bboxCoords.top]);
  if (!item.parent) return null;
  if (!item.inViewPort) return null;

  return (
    <Fragment>
      <Ellipse
        x={item.canvasX}
        y={item.canvasY}
        ref={el => setShapeRef(el)}
        radiusX={item.canvasRadiusX}
        radiusY={item.canvasRadiusY}
        fill={regionStyles.fillColor}
        stroke={regionStyles.strokeColor}
        strokeWidth={regionStyles.strokeWidth}
        strokeScaleEnabled={false}
        perfectDrawEnabled={false}
        shadowForStrokeEnabled={false}
        shadowBlur={0}
        scaleX={item.scaleX}
        scaleY={item.scaleY}
        opacity={item.isDrawing ? 1 : 0}
        rotation={item.rotation}
        name={`${item.id} _transformable`}
        onTransform={({ target }) => {
          // resetting the skew makes transformations weird but predictable
          target.setAttr("skewX", 0);
          target.setAttr("skewY", 0);
        }}
        onTransformEnd={e => {
          const t = e.target;
          item.setPosition(
            t.getAttr("x"),
            t.getAttr("y"),
            t.getAttr("radiusX") * t.getAttr("scaleX"),
            t.getAttr("radiusY") * t.getAttr("scaleY"),
            t.getAttr("rotation")
          );

          t.setAttr("scaleX", 1);
          t.setAttr("scaleY", 1);
          item.notifyDrawingFinished();
        }}
        onDragStart={e => {
          if (item.parent.getSkipInteractions()) {
            e.currentTarget.stopDrag(e.evt);
            return;
          }
          item.annotation.history.freeze(item.id);
        }}
        onDragEnd={e => {
          const t = e.target;

          item.setPosition(
            t.getAttr("x"),
            t.getAttr("y"),
            t.getAttr("radiusX"),
            t.getAttr("radiusY"),
            t.getAttr("rotation")
          );
          item.setScale(t.getAttr("scaleX"), t.getAttr("scaleY"));
          item.annotation.history.unfreeze(item.id);
          item.notifyDrawingFinished();
        }}
        dragBoundFunc={createDragBoundFunc(item, {
          x: item.x - item.bboxCoords.left,
          y: item.y - item.bboxCoords.top
        })}
        onMouseOver={() => {
          if (store.annotationStore.selected.relationMode) {
            item.setHighlight(true);
            stage.container().style.cursor = Constants.RELATION_MODE_CURSOR;
          } else {
            stage.container().style.cursor = Constants.POINTER_CURSOR;
          }
        }}
        onMouseOut={() => {
          stage.container().style.cursor = Constants.DEFAULT_CURSOR;

          if (store.annotationStore.selected.relationMode) {
            item.setHighlight(false);
          }
        }}
        onClick={e => {
          if (item.parent.getSkipInteractions()) return;

          if (store.annotationStore.selected.relationMode) {
            stage.container().style.cursor = Constants.DEFAULT_CURSOR;
          }

          item.setHighlight(false);
          item.onClickRegion(e);
        }}
        draggable={!item.isReadOnly()}
        listening={!suggestion}
      />
      <LabelOnEllipse
        item={item}
        color={regionStyles.strokeColor}
        strokewidth={regionStyles.strokeWidth}
      />
      {!item.isDrawing && (
        <Group
          key={item.id ? item.id : guidGenerator(5)}
          name={item.id}
          ref={el => setShapeRef(el)}
          onMouseOver={() => {
            if (store.annotationStore.selected.relationMode) {
              item.setHighlight(true);
              stage.container().style.cursor = Constants.RELATION_MODE_CURSOR;
            } else {
              stage.container().style.cursor = Constants.POINTER_CURSOR;
            }
          }}
          onMouseOut={() => {
            stage.container().style.cursor = Constants.DEFAULT_CURSOR;

            if (store.annotationStore.selected.relationMode) {
              item.setHighlight(false);
            }
          }}
          onClick={e => {
            // create regions over another regions with Cmd/Ctrl pressed
            if (item.parent.getSkipInteractions()) return;
            if (item.isDrawing) return;

            e.cancelBubble = true;

            if (!item.closed) return;

            if (store.annotationStore.selected.relationMode) {
              stage.container().style.cursor = Constants.DEFAULT_CURSOR;
            }

            item.setHighlight(false);
            item.onClickRegion(e);
          }}
          {...dragProps}
          draggable={
            !item.isReadOnly() &&
            (!item.inSelection || item.parent?.selectedRegions?.length === 1)
          }
          listening={!suggestion}
        >
          <LabelOnPolygon item={item} color={regionStyles.strokeColor} />

          {item.mouseOverStartPoint}

          {item.polygonPoints && item.closed ? (
            <Poly
              item={item}
              colors={regionStyles}
              dragProps={dragProps}
              draggable={
                !item.isReadOnly() &&
                item.inSelection &&
                item.parent?.selectedRegions?.length > 1
              }
            />
          ) : null}
          {item.polygonPoints && !item.isReadOnly() ? (
            <Edges item={item} regionStyles={regionStyles} />
          ) : null}
          {item.polygonPoints && !item.isReadOnly()
            ? renderCircles(item.polygonPoints)
            : null}
        </Group>
      )}
    </Fragment>
  );
};

const HtxEllipseBasedPolygon = AliveRegion(HtxEllipseBasedPolygonView);

const detector = value => {
  if (!value) return false;
  const keys = Object.keys(value);
  return (
    keys.includes("polygonPoints") &&
    keys.includes("radiusX") &&
    keys.includes("radiusY")
  );
};

Registry.addTag(
  "ellipsebasedpolygonregion",
  EllipseBasedPolygonRegionModel,
  HtxEllipseBasedPolygon
);
Registry.addRegionType(EllipseBasedPolygonRegionModel, "image", detector);

export { EllipseBasedPolygonRegionModel, HtxEllipseBasedPolygon };
