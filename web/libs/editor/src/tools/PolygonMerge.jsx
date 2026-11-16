import { observer } from "mobx-react";
import { types } from "mobx-state-tree";

import BaseTool from "./Base";
import ToolMixin from "../mixins/Tool";
import { Tool } from "../components/Toolbar/Tool";
import InfoModal from "../components/Infomodal/Infomodal";
import { IconMergeTool } from "../assets/icons";
import { mergePolygons } from "../utils/polygonOps";

const labelKeys = (value) => Object.keys(value).filter((k) => k.endsWith("labels"));

// One string per region describing all its labels, to tell whether merged regions agree
const labelSignature = (results) =>
  results
    .map((res) => `${res.from_name}:${labelKeys(res.value).map((k) => [...res.value[k]].sort().join("|"))}`)
    .sort()
    .join(";");

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
      const regions = self.annotation?.selectedRegions ?? [];
      const ok =
        regions.length > 1 &&
        regions.every(
          (r) =>
            r.type === "polygonregion" &&
            r.closed &&
            !r.isReadOnly() &&
            r.parent?.name === self.obj?.name &&
            r.results.some((res) => res.from_name === self.control),
        );

      return ok ? regions : null;
    },
  }))
  .actions((self) => ({
    merge() {
      // the hotkey fires even when the button is disabled
      const regions = self.mergeRegions;

      if (!regions) return;

      const serialized = regions.map((r) => r.results.map((res) => res.serialize()).filter(Boolean));

      if (serialized.some((results) => !results.length)) return;
      const merged = mergePolygons(
        serialized.map(([res]) => ({ points: res.value.points, holes: res.value.holes?.[0] })),
      );

      if (merged.error) {
        InfoModal.warning(merged.error);
        return;
      }

      const sameLabels = new Set(serialized.map(labelSignature)).size === 1;
      // shortcut: per-region results (textarea/choices) are dropped, carry them over if a project needs them
      const results = serialized[0]
        .filter((res) => res.type === "polygon" || res.type.endsWith("labels"))
        .map((res) => {
          const { ring, outerclosed, holes, ...value } = res.value;

          value.points = merged.points;
          if (merged.holes.length) Object.assign(value, { ring: true, outerclosed: true, holes: [merged.holes] });
          // different labels: leave the merged region unlabeled so the user picks a label for it
          if (!sameLabels) labelKeys(value).forEach((k) => (value[k] = []));
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
