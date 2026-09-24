import Konva from "konva";
import React, { memo, useContext, useEffect, useMemo } from "react";
import { Group, Line, Shape } from "react-konva";
import { destroy, detach, getRoot, isAlive, types } from "mobx-state-tree";

import Constants from "../core/Constants";
import NormalizationMixin from "../mixins/Normalization";
import RegionsMixin from "../mixins/Regions";
import Registry from "../core/Registry";
import { ImageModel } from "../tags/object/Image";
import { LabelOnPolygon } from "../components/ImageView/LabelOnRegion";
import { PolygonPoint, PolygonPointView } from "./PolygonPoint";
import { green } from "@ant-design/colors";
import { guidGenerator } from "../core/Helpers";
import { AreaMixin } from "../mixins/AreaMixin";
import { useRegionStyles } from "../hooks/useRegionColor";
import { AliveRegion } from "./AliveRegion";
import { KonvaRegionMixin } from "../mixins/KonvaRegion";
import { observer } from "mobx-react";
import { createDragBoundFunc } from "../utils/image";
import { ImageViewContext } from "../components/ImageView/ImageViewContext";
import { FF_DEV_2432, FF_DEV_3793, isFF } from "../utils/feature-flags";
import { fixMobxObserve } from "../utils/utilities";
import { RELATIVE_STAGE_HEIGHT, RELATIVE_STAGE_WIDTH } from "../components/ImageView/Image";

const PolygonRegionAbsoluteCoordsDEV3793 = types
  .model({
    coordstype: types.optional(types.enumeration(["px", "perc"]), "perc"),
  })
  .actions((self) => ({
    updateImageSize(wp, hp, sw, sh) {
      if (self.coordstype === "px") {
        [...self.points, ...self.holes].forEach((p) => {
          const x = (sw * p.relativeX) / RELATIVE_STAGE_WIDTH;
          const y = (sh * p.relativeY) / RELATIVE_STAGE_HEIGHT;

          p._setPos(x, y);
        });
      }

      if (!self.annotation.sentUserGenerate && self.coordstype === "perc") {
        [...self.points, ...self.holes].forEach((p) => {
          const x = (sw * p.x) / RELATIVE_STAGE_WIDTH;
          const y = (sh * p.y) / RELATIVE_STAGE_HEIGHT;

          self.coordstype = "px";
          p._setPos(x, y);
        });
      }
    },
  }));

const Model = types
  .model({
    id: types.optional(types.identifier, guidGenerator),
    pid: types.optional(types.string, guidGenerator),
    type: "polygonregion",
    object: types.late(() => types.reference(ImageModel)),

    points: types.array(types.union(PolygonPoint, types.array(types.number)), []),
    holes: types.array(types.union(PolygonPoint, types.array(types.number)), []),
    ring: false,
    outerclosed: false,
    closed: true,
  })
  .volatile(() => ({
    mouseOverStartPoint: false,
    selectedPoint: null,
    drawingPoint: null,
    hideable: true,
    _supportsTransform: true,
    useTransformer: true,
    preferTransformer: false,
    supportsRotate: false,
    supportsScale: true,
  }))
  .views((self) => ({
    get store() {
      return getRoot(self);
    },
    get bboxCoords() {
      if (!self.points?.length || !isAlive(self)) return {};

      const bbox = self.points.reduce(
        (bboxCoords, point) => ({
          left: Math.min(bboxCoords.left, point.x),
          top: Math.min(bboxCoords.top, point.y),
          right: Math.max(bboxCoords.right, point.x),
          bottom: Math.max(bboxCoords.bottom, point.y),
        }),
        {
          left: self.points[0].x,
          top: self.points[0].y,
          right: self.points[0].x,
          bottom: self.points[0].y,
        },
      );

      if (!isFF(FF_DEV_3793)) {
        // recalc on resize
        fixMobxObserve(self.parent.stageWidth, self.parent.stageHeight);
      }

      return bbox;
    },
    get flattenedPoints() {
      return getFlattenedPoints(this.points);
    },
    get flattenedDrawingPoints() {
      if (!self.drawingPoint) return this.flattenedPoints;

      return [
        ...this.flattenedPoints,
        self.parent.internalToCanvasX(self.drawingPoint.x),
        self.parent.internalToCanvasY(self.drawingPoint.y),
      ];
    },
    get flattenedHolePoints() {
      return getFlattenedPoints(self.holes);
    },
    get flattenedHoleDrawingPoints() {
      if (!self.drawingPoint || !self.outerclosed) return this.flattenedHolePoints;
      return [
        ...this.flattenedHolePoints,
        self.parent.internalToCanvasX(self.drawingPoint.x),
        self.parent.internalToCanvasY(self.drawingPoint.y),
      ];
    },
    get activePoints() {
      return self.ring && self.outerclosed ? self.holes : self.points;
    },
  }))
  .actions((self) => {
    return {
      afterCreate() {
        if (!self.points.length) return;
        if (!self.points[0].id) {
          self.points = self.points.map(([x, y], index) => ({
            id: guidGenerator(),
            x,
            y,
            size: self.pointSize,
            style: self.pointStyle,
            index,
          }));
        }
        if (self.holes.length && !self.holes[0].id) {
          self.holes = self.holes.map(([x, y], index) => ({
            id: guidGenerator(),
            x,
            y,
            size: self.pointSize,
            style: self.pointStyle,
            index,
          }));
        }
        if (self.ring) {
          self.outerclosed = self.outerclosed || self.holes.length > 0;
          self.useTransformer = false;
          self._supportsTransform = false;
        }
        if (!isFF(FF_DEV_2432)) self.closed = self.points.length > 2;
        self.checkSizes();
      },

      /**
       * @todo excess method; better to handle click only on start point
       * Handler for mouse on start point of Polygon
       * @param {boolean} val
       */
      setMouseOverStartPoint(value) {
        self.mouseOverStartPoint = value;
      },

      setDrawingPoint(x, y) {
        self.drawingPoint = { x, y };
      },

      // @todo not used
      setSelectedPoint(point) {
        if (self.selectedPoint) {
          self.selectedPoint.selected = false;
        }

        point.selected = true;
        self.selectedPoint = point;
      },

      handleMouseMove({ e, flattenedPoints }) {
        const { offsetX, offsetY } = e.evt;
        const [cursorX, cursorY] = self.parent.fixZoomedCoords([offsetX, offsetY]);
        const [x, y] = getAnchorPoint({ flattenedPoints, cursorX, cursorY });

        const group = e.currentTarget;
        const layer = e.currentTarget.getLayer();
        const zoom = self.parent.zoomScale;

        moveHoverAnchor({ point: [x, y], group, layer, zoom });
      },

      handleMouseLeave({ e }) {
        removeHoverAnchor({ layer: e.currentTarget.getLayer() });
      },

      handleLineClick({ e, flattenedPoints, insertIdx, contour = "outer" }) {
        if (!self.closed || !self.selected) return;

        e.cancelBubble = true;

        removeHoverAnchor({ layer: e.currentTarget.getLayer() });

        const { offsetX, offsetY } = e.evt;

        const [cursorX, cursorY] = self.parent.fixZoomedCoords([offsetX, offsetY]);
        const point = getAnchorPoint({ flattenedPoints, cursorX, cursorY });

        self.insertPoint(insertIdx, point[0], point[1], contour);
      },

      deletePoint(point) {
        const contour = self.getContourForPoint(point);
        const willNotEliminateClosedShape = contour.length <= 3 && point.parent.closed;
        const isLastPoint = contour.length === 1;
        const isSelected = self.selectedPoint === point;

        if (willNotEliminateClosedShape || isLastPoint) return;
        if (isSelected) self.selectedPoint = null;
        destroy(point);
      },

      addPoint(x, y) {
        if (self.closed) return;

        const point = self.control?.getSnappedPoint({ x, y });

        self._addPoint(point.x, point.y);
      },

      getContourForPoint(point) {
        return self.holes.includes(point) ? self.holes : self.points;
      },

      isContourClosed(point) {
        return self.getContourForPoint(point) === self.points ? self.outerclosed || self.closed : self.closed;
      },

      startInner(x, y) {
        if (!self.ring || !self.outerclosed || self.holes.length) return;
        const point = self.control?.getSnappedPoint({ x, y }) ?? { x, y };
        self.drawingPoint = null;
        self.holes.push({
          id: guidGenerator(),
          x: point.x,
          y: point.y,
          size: self.pointSize,
          style: self.pointStyle,
          index: 0,
        });
      },

      addRingPoint(x, y) {
        if (!self.ring || self.closed) return;
        const contour = self.activePoints;
        const point = self.control?.getSnappedPoint({ x, y }) ?? { x, y };
        if (contour.length && self.parent.isSamePixel(contour[0], point)) {
          self.closeActiveContour();
          return;
        }
        contour.push({
          id: guidGenerator(),
          x: point.x,
          y: point.y,
          size: self.pointSize,
          style: self.pointStyle,
          index: contour.length,
        });
      },

      closeOuter() {
        if (!self.ring || self.outerclosed || self.points.length < 3) return;
        self.drawingPoint = null;
        self.outerclosed = true;
      },

      closeInner() {
        if (!self.ring || self.closed || self.holes.length < 3) return;
        self.drawingPoint = null;
        self.closed = true;
      },

      closeActiveContour() {
        if (!self.outerclosed) self.closeOuter();
        else self.closeInner();
      },

      closeContour(point) {
        if (self.ring && !self.closed) self.closeActiveContour();
        else self.closePoly();
      },

      setPoints(points) {
        self.points.forEach((p, idx) => {
          p.x = points[idx * 2];
          p.y = points[idx * 2 + 1];
        });
      },

      insertPoint(insertIdx, x, y, contour = "outer") {
        const targetPoints = contour === "inner" ? self.holes : self.points;
        const pointCoords = self.control?.getSnappedPoint({
          x: self.parent.canvasToInternalX(x),
          y: self.parent.canvasToInternalY(y),
        });
        const isMatchWithPrevPoint =
          targetPoints[insertIdx - 1] && self.parent.isSamePixel(pointCoords, targetPoints[insertIdx - 1]);
        const isMatchWithNextPoint =
          targetPoints[insertIdx] && self.parent.isSamePixel(pointCoords, targetPoints[insertIdx]);

        if (isMatchWithPrevPoint || isMatchWithNextPoint) {
          return;
        }

        const p = {
          id: guidGenerator(),
          x: pointCoords.x,
          y: pointCoords.y,
          size: self.pointSize,
          style: self.pointStyle,
          index: targetPoints.length,
        };

        targetPoints.splice(insertIdx, 0, p);

        return targetPoints[insertIdx];
      },

      _addPoint(x, y) {
        const firstPoint = self.points[0];

        // This is mostly for "snap to pixel" mode,
        // 'cause there is also an ability to close polygon by clicking on the first point precisely
        if (self.parent.isSamePixel(firstPoint, { x, y })) {
          self.closePoly();
          return;
        }

        self.points.push({
          id: guidGenerator(),
          x,
          y,
          size: self.pointSize,
          style: self.pointStyle,
          index: self.points.length,
        });
      },

      closePoly() {
        if (self.closed || self.points.length < 3) return;
        self.drawingPoint = null;
        self.closed = true;
      },

      canClose(x, y) {
        if (self.points.length < 2) return false;

        const p1 = self.points[0];
        const p2 = { x, y };

        const r = 50;
        const dist_points = (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2;

        if (dist_points < r) {
          return true;
        }
        return false;
      },

      destroyRegion() {
        detach(self.points);
        destroy(self.points);
        detach(self.holes);
        destroy(self.holes);
      },

      afterUnselectRegion() {
        if (self.selectedPoint) {
          self.selectedPoint.selected = false;
        }

        // self.points.forEach(p => p.computeOffset());
      },

      setScale(x, y) {
        self.scaleX = x;
        self.scaleY = y;
      },

      updateImageSize() {},

      /**
       * @example
       * {
       *   "original_width": 1920,
       *   "original_height": 1280,
       *   "image_rotation": 0,
       *   "value": {
       *     "points": [[2, 2], [3.5, 8.1], [3.5, 12.6]],
       *     "polygonlabels": ["Car"]
       *   }
       * }
       * @typedef {Object} PolygonRegionResult
       * @property {number} original_width width of the original image (px)
       * @property {number} original_height height of the original image (px)
       * @property {number} image_rotation rotation degree of the image (deg)
       * @property {Object} value
       * @property {number[][]} value.points list of (x, y) coordinates of the polygon by percentage of the image size (0-100)
       */

      /**
       * @return {PolygonRegionResult}
       */
      serialize() {
        if (!isFF(FF_DEV_2432) && self.points.length < 3) return null;
        const value = {
          points: isFF(FF_DEV_3793)
            ? self.points.map((p) => [p.x, p.y])
            : self.points.map((p) => [self.convertXToPerc(p.x), self.convertYToPerc(p.y)]),
          ...(isFF(FF_DEV_2432) ? { closed: self.closed } : {}),
          ...(self.ring ? {
            ring: true,
            outerclosed: self.outerclosed,
            holes: [isFF(FF_DEV_3793)
              ? self.holes.map((p) => [p.x, p.y])
              : self.holes.map((p) => [self.convertXToPerc(p.x), self.convertYToPerc(p.y)])],
          } : {}),
        };

        return self.parent.createSerializedResult(self, value);
      },
    };
  });

const PolygonRegionModel = types.compose(
  "PolygonRegionModel",
  RegionsMixin,
  AreaMixin,
  NormalizationMixin,
  KonvaRegionMixin,
  Model,
  ...(isFF(FF_DEV_3793) ? [] : [PolygonRegionAbsoluteCoordsDEV3793]),
).preProcessSnapshot((snapshot) => ({
  ...snapshot,
  holes: Array.isArray(snapshot.holes?.[0]) && Array.isArray(snapshot.holes[0]?.[0])
    ? snapshot.holes[0]
    : snapshot.holes ?? [],
  ring: snapshot.ring || Boolean(snapshot.holes?.length),
}));

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
    ((point2Y - point1Y) * (point2Y - point1Y) + (point2X - point1X) * (point2X - point1X));
  const x =
    cursorX -
    ((point2Y - point1Y) *
      (point2X * point1Y - point1X * point2Y + cursorX * (point2Y - point1Y) - cursorY * (point2X - point1X))) /
      ((point2Y - point1Y) * (point2Y - point1Y) + (point2X - point1X) * (point2X - point1X));

  return [x, y];
}

function getFlattenedPoints(points) {
  const p = points.map((p) => [p.canvasX, p.canvasY]);

  return p.reduce((flattenedPoints, point) => flattenedPoints.concat(point), []);
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
    radius: 5,
  });

  group.add(hoverAnchor);
  layer.draw();
  return hoverAnchor;
}

function moveHoverAnchor({ point, group, layer, zoom }) {
  const hoverAnchor = getHoverAnchor({ layer }) || createHoverAnchor({ point, group, layer, zoom });

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
          onTransformEnd={(e) => {
            if (e.target !== e.currentTarget) return;

            const t = e.target;

            const d = [t.getAttr("x", 0), t.getAttr("y", 0)];
            const scale = [t.getAttr("scaleX", 1), t.getAttr("scaleY", 1)];
            const points = t.getAttr("points");

            item.setPoints(
              points.reduce((result, coord, idx) => {
                const isXCoord = idx % 2 === 0;

                if (isXCoord) {
                  const point = item.control?.getSnappedPoint({
                    x: item.parent.canvasToInternalX(coord * scale[0] + d[0]),
                    y: item.parent.canvasToInternalY(points[idx + 1] * scale[1] + d[1]),
                  });

                  result.push(point.x, point.y);
                }
                return result;
              }, []),
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
  }),
);

const signedArea = (points) => {
  let area = 0;
  for (let index = 0; index < points.length; index += 2) {
    const next = (index + 2) % points.length;
    area += points[index] * points[next + 1] - points[next] * points[index + 1];
  }
  return area / 2;
};

const traceContour = (context, points) => {
  if (points.length < 6) return;
  context.moveTo(points[0], points[1]);
  for (let index = 2; index < points.length; index += 2) {
    context.lineTo(points[index], points[index + 1]);
  }
  context.closePath();
};

const reverseContour = (points) => {
  const reversed = [];
  for (let index = points.length - 2; index >= 0; index -= 2) {
    reversed.push(points[index], points[index + 1]);
  }
  return reversed;
};

const RingPoly = observer(({ item, colors }) => {
  const outer = item.flattenedPoints;
  const inner = item.flattenedHolePoints;

  return (
    <Shape
      name="ring-polygon"
      fill={colors.fillColor}
      stroke={colors.strokeColor}
      strokeWidth={colors.strokeWidth}
      strokeScaleEnabled={false}
      perfectDrawEnabled={false}
      shadowForStrokeEnabled={false}
      sceneFunc={(context, shape) => {
        context.beginPath();
        traceContour(context, outer);
        if (inner.length >= 6) {
          traceContour(context, signedArea(outer) * signedArea(inner) > 0 ? reverseContour(inner) : inner);
        }
        context.fillStrokeShape(shape);
      }}
    />
  );
});

const DrawingPreview = observer(({ item, colors }) => {
  const points = item.ring && item.outerclosed ? item.flattenedHoleDrawingPoints : item.flattenedDrawingPoints;

  if (!item.drawingPoint || points.length < 4) return null;

  return (
    <Line
      name="polygon-drawing-preview"
      lineJoin="round"
      lineCap="square"
      stroke={colors.strokeColor}
      strokeWidth={colors.strokeWidth}
      strokeScaleEnabled={false}
      perfectDrawEnabled={false}
      shadowForStrokeEnabled={false}
      points={points}
      fill={colors.fillColor}
      closed={points.length >= 6}
      listening={false}
    />
  );
});

/**
 * Line between 2 points
 */ 
const Edge = observer(({ name, item, idx, p1, p2, closed, regionStyles, contour }) => {
  const insertIdx = idx + 1; // idx1 + 1 or idx2
  const flattenedPoints = [p1.canvasX, p1.canvasY, p2.canvasX, p2.canvasY];

  const lineProps = closed
    ? {
        stroke: "transparent",
        strokeWidth: regionStyles.strokeWidth,
        strokeScaleEnabled: false,
      }
    : {
        stroke: regionStyles.strokeColor,
        strokeWidth: regionStyles.strokeWidth,
        strokeScaleEnabled: false,
      };

  return (
    <Group
      key={name}
      name={name}
      onClick={(e) => item.handleLineClick({ e, flattenedPoints, insertIdx, contour })}
      onMouseMove={(e) => {
        if (!item.closed || !item.selected || item.isReadOnly()) return;

        item.handleMouseMove({ e, flattenedPoints });
      }}
      onMouseLeave={(e) => item.handleMouseLeave({ e })}
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
  observer(({ item, regionStyles, points = item.points, closed = item.closed, contour = "outer" }) => {
    const name = `borders-${contour}`;

    if (item.closed && (item.parent.useTransformer || !item.selected)) {
      return null;
    }
    return (
      <Group key={name} name={name}>
        {points.map((p, idx) => {
          const idx1 = idx;
          const idx2 = idx === points.length - 1 ? 0 : idx + 1;

          if (!closed && idx2 === 0) {
            return null;
          }

          return (
            <Edge
              key={`border_${idx1}_${idx2}`}
              name={`border_${idx1}_${idx2}`}
              item={item}
              idx={idx1}
              p1={points[idx]}
              p2={points[idx2]}
              closed={closed}
              contour={contour}
              regionStyles={regionStyles}
            />
          );
        })}
      </Group>
    );
  }),
);

const HtxPolygonView = ({ item, setShapeRef }) => {
  const { store } = item;
  const { suggestion } = useContext(ImageViewContext) ?? {};

  const regionStyles = useRegionStyles(item, {
    useStrokeAsFill: true,
  });

  function renderCircle({ points, idx, contour }) {
    const name = `anchor_${contour}_${points.length}_${idx}`;
    const point = points[idx];

    if (!item.closed || (item.closed && item.selected)) {
      return <PolygonPointView item={point} name={name} key={name} />;
    }
  }

  function renderCircles(points, contour) {
    const name = `anchors-${contour}`;

    if (item.closed && (item.parent.useTransformer || !item.selected)) {
      return null;
    }
    return (
      <Group key={name} name={name}>
        {points.map((p, idx) => renderCircle({ points, idx, contour }))}
      </Group>
    );
  }

  const dragProps = useMemo(() => {
    let isDragging = false;

    return {
      onDragStart: (e) => {
        if (e.target !== e.currentTarget) return;
        if (item.parent.getSkipInteractions()) {
          e.currentTarget.stopDrag(e.evt);
          return;
        }
        isDragging = true;
        item.annotation.setDragMode(true);

        item.annotation.history.freeze(item.id);
      },
      dragBoundFunc: createDragBoundFunc(item, { x: -item.bboxCoords.left, y: -item.bboxCoords.top }),
      onDragEnd: (e) => {
        if (!isDragging) return;
        const t = e.target;

        if (e.target === e.currentTarget) {
          item.annotation.setDragMode(false);

          const point = item.control?.getSnappedPoint({
            x: item.parent?.canvasToInternalX(t.getAttr("x")),
            y: item.parent?.canvasToInternalY(t.getAttr("y")),
          });

          point.x = item.parent?.internalToCanvasX(point.x);
          point.y = item.parent?.internalToCanvasY(point.y);

          [...item.points, ...item.holes].forEach((p) => p.movePoint(point.x, point.y));
          item.annotation.history.unfreeze(item.id);
        }

        t.setAttr("x", 0);
        t.setAttr("y", 0);
        isDragging = false;
      },
    };
  }, [item.bboxCoords.left, item.bboxCoords.top]);

  useEffect(() => {
    if (isFF(FF_DEV_2432) && !item.closed) item.control.tools.Polygon.resumeUnfinishedRegion(item);
  }, [item.closed]);

  if (!item.parent) return null;
  if (!item.inViewPort) return null;

  const stage = item.parent?.stageRef;
  const selectedTool = item.parent?.getToolsManager().findSelectedTool();
  const isCreatingPolygon = item.annotation?.isDrawing
    && ["PolygonTool", "RingPolygonTool", "OpenCVPolygonTool"].includes(selectedTool?.toolName);

  return (
    <Group
      key={item.id ? item.id : guidGenerator(5)}
      name={item.id}
      ref={(el) => setShapeRef(el)}
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
      onClick={(e) => {
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
      draggable={!item.isReadOnly() && (!item.inSelection || item.parent?.selectedRegions?.length === 1)}
      listening={!suggestion && !(isCreatingPolygon && item.closed)}
    >
      <LabelOnPolygon item={item} color={regionStyles.strokeColor} />

      {item.mouseOverStartPoint}

      {item.ring && item.outerclosed ? (
        <RingPoly item={item} colors={regionStyles} />
      ) : item.points && item.closed ? (
        <Poly
          item={item}
          colors={regionStyles}
          dragProps={dragProps}
          draggable={!item.isReadOnly() && item.inSelection && item.parent?.selectedRegions?.length > 1}
        />
      ) : null}
      {!item.closed ? <DrawingPreview item={item} colors={regionStyles} /> : null}
      {item.points && !item.isReadOnly() ? (
        <Edges
          item={item}
          regionStyles={regionStyles}
          closed={item.ring ? item.outerclosed : item.closed}
        />
      ) : null}
      {item.ring && item.holes.length && !item.isReadOnly() ? (
        <Edges item={item} regionStyles={regionStyles} points={item.holes} closed={item.closed} contour="inner" />
      ) : null}
      {item.points && !item.isReadOnly() ? renderCircles(item.points, "outer") : null}
      {item.ring && item.holes.length && !item.isReadOnly() ? renderCircles(item.holes, "inner") : null}
    </Group>
  );
};

const HtxPolygon = AliveRegion(HtxPolygonView);

Registry.addTag("polygonregion", PolygonRegionModel, HtxPolygon);
Registry.addRegionType(PolygonRegionModel, "image", (value) => !!value.points);

export { PolygonRegionModel, HtxPolygon };
