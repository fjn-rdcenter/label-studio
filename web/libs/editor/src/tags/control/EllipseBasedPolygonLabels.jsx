import React from "react";
import { observer } from "mobx-react";
import { types } from "mobx-state-tree";

import LabelMixin from "../../mixins/LabelMixin";
import Registry from "../../core/Registry";
import SelectedModelMixin from "../../mixins/SelectedModel";
import Types from "../../core/Types";
import { HtxLabels, LabelsModel } from "./Labels/Labels";
import { EllipseBasedPolygonModel } from "./EllipseBasedPolygon";
import ControlBase from "./Base";

/**
 * The `EllipseBasedPolygonLabels` tag is used to create labeled polygons from ellipse. Use to apply labels to polygons in semantic segmentation tasks.
 *
 * Use with the following data types: image.
 * @example
 * <!--Basic labeling configuration for polygonal semantic segmentation of images -->
 * <View>
 *   <Image name="image" value="$image" />
 *   <EllipseBasedPolygonLabels name="labels" toName="image">
 *     <Label value="Car" />
 *     <Label value="Sign" />
 *   </EllipseBasedPolygonLabels>
 * </View>
 * @name EllipseBasedPolygonLabels
 * @regions PolygonRegion
 * @meta_title Polygon Label Tag for Labeling Polygons in Images
 * @meta_description Customize Label Studio with the EllipseBasedPolygonLabels tag and label polygons in images for semantic segmentation machine learning and data science projects.
 * @param {string} name                             - Name of tag
 * @param {string} toName                           - Name of image to label
 * @param {single|multiple=} [choice=single]        - Configure whether you can select one or multiple labels
 * @param {number} [maxUsages]                      - Maximum number of times a label can be used per task
 * @param {boolean} [showInline=true]               - Show labels in the same visual line
 * @param {number} [opacity=0.2]                    - Opacity of polygon
 * @param {string} [fillColor]                      - Polygon fill color in hexadecimal
 * @param {string} [strokeColor]                    - Stroke color in hexadecimal
 * @param {number} [strokeWidth=1]                  - Width of stroke
 * @param {small|medium|large} [pointSize=medium]   - Size of polygon handle points
 * @param {rectangle|circle} [pointStyle=rectangle] - Style of points
 * @param {pixel|none} [snap=none]                  - Snap polygon to image pixels
 */

const Validation = types.model({
  controlledTags: Types.unionTag(["Image"]),
});

const ModelAttrs = types.model("EllipseBasedPolygonLabelsModel", {
  type: "ellipsebasedpolygonlabels",
  children: Types.unionArray(["label", "header", "view", "hypertext"]),
});

const Composition = types.compose(
  ControlBase,
  LabelsModel,
  ModelAttrs,
  EllipseBasedPolygonModel,
  Validation,
  LabelMixin,
  SelectedModelMixin.props({ _child: "LabelModel" }),
);

const EllipseBasedPolygonLabelsModel = types.compose("EllipseBasedPolygonLabelsModel", Composition);

const HtxPolygonLabels = observer(({ item }) => {
  return <HtxLabels item={item} />;
});

Registry.addTag("ellipsebasedpolygonlabels", EllipseBasedPolygonLabelsModel, HtxPolygonLabels);

export { HtxPolygonLabels, EllipseBasedPolygonLabelsModel };
