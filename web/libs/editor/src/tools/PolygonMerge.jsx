import { observer } from "mobx-react";
import { types } from "mobx-state-tree";

import BaseTool from "./Base";
import ToolMixin from "../mixins/Tool";
import { Tool } from "../components/Toolbar/Tool";
import InfoModal from "../components/Infomodal/Infomodal";
import { IconMergeTool } from "../assets/icons";
import { mergePolygons, withGeometry } from "../utils/polygonOps";

// vertices of a smaller polygon within this many screen px of the largest one are dropped
const BLEND_PX = 8;
// slits narrower than this many canvas px (not screen: they are data gaps, ~1 image px) get closed
const SLIT_PX = 3;

const labelKeys = (value) => Object.keys(value).filter((k) => k.endsWith("labels"));

// One string per region describing all its labels, to tell whether merged regions agree
const labelSignature = (results) =>
  results
    .map((res) => `${res.from_name}:${labelKeys(res.value).map((k) => [...res.value[k]].sort().join("|"))}`)
    .sort()
    .join(";");

// Is r a closed, editable polygon of this tool's control and image
export const editablePolygon = (tool, r) =>
  r?.type === "polygonregion" &&
  r.closed &&
  !r.isReadOnly() &&
  r.parent?.name === tool.obj?.name &&
  r.results.some((res) => res.from_name === tool.control);

// Two or more selected, closed, editable polygons of this tool's control and image, or null
export const selectedPolygons = (tool) => {
  const regions = tool.annotation?.selectedRegions ?? [];

  return regions.length > 1 && regions.every((r) => editablePolygon(tool, r)) ? regions : null;
};

const toCanvas = (points) => points.map((p) => [p.canvasX, p.canvasY]);

// Contours of a polygon region in canvas coords
export const regionGeometry = (r) => ({ points: toCanvas(r.points), holes: toCanvas(r.holes) });

const ToolView = observer(({ item }) => (
  <Tool
    label="Gộp polygon"
    ariaLabel="polygon-merge"
    shortcut={item.shortcut}
    icon={<IconMergeTool />}
    tool={item}
    disabled={!item.mergeRegions}
    onClick={() => item.merge()}
  />
));

const _Tool = types
  .model("PolygonMergeTool", {
    group: "segmentation",
    shortcut: "G",
  })
  .views((self) => ({
    get viewClass() {
      return () => <ToolView item={self} />;
    },

    // Selected polygons this tool may merge, or null
    get mergeRegions() {
      return selectedPolygons(self);
    },
  }))
  .actions((self) => ({
    merge() {
      // the hotkey fires even when the button is disabled
      const regions = self.mergeRegions;

      if (!regions) return;

      const serialized = regions.map((r) => r.results.map((res) => res.serialize()).filter(Boolean));

      if (serialized.some((results) => !results.length)) return;
      // canvas coords, so the blend tolerance is in screen px
      const merged = mergePolygons(regions.map(regionGeometry), BLEND_PX / (self.obj?.zoomScale || 1), SLIT_PX);

      if (merged.error) {
        InfoModal.warning(merged.error);
        return;
      }

      const { bigIndex, absorbed } = merged;
      const region = regions[bigIndex];
      const toPerc = (ring) => ring.map(([x, y]) => [region.convertXToPerc(x), region.convertYToPerc(y)]);
      const geometry = { points: toPerc(merged.points), holes: toPerc(merged.holes) };
      // labels differ and small polygons don't sit inside the big one: let the user pick a label
      const resetLabels = !absorbed && new Set(serialized.map(labelSignature)).size > 1;
      // shortcut: per-region results (textarea/choices) are dropped, carry them over if a project needs them
      const results = serialized[bigIndex]
        .filter((res) => res.type === "polygon" || res.type.endsWith("labels"))
        .map((res) => {
          const value = withGeometry(res.value, geometry);

          if (resetLabels) labelKeys(value).forEach((k) => (value[k] = []));
          return { ...res, value };
        });

      const { annotation } = self;

      annotation.history.freeze("polygon-merge");
      const created = annotation.appendResults(results);

      // appendResults bails on read-only annotations; never delete the originals then
      if (created?.length) regions.forEach((r) => r.deleteRegion());
      annotation.history.unfreeze("polygon-merge");
      if (created?.length) annotation.selectAreas(created);
    },
  }));

const PolygonMerge = types.compose(_Tool.name, ToolMixin, BaseTool, _Tool);

export { PolygonMerge };
