import chroma from "chroma-js";
import type { FC } from "react";
import { observer } from "mobx-react";

import { Block, Elem } from "../../../utils/bem";

export const RegionLabels: FC<{ region: LSFRegion }> = observer(({ region }) => {
  const labelsInResults = region.labelings.map((result: any) => result.selectedLabels || []);
  const labels: any[] = [].concat(...labelsInResults);

  if (!labels.length) return <Block name="labels-list">No label</Block>;

  return (
    <Block name="labels-list">
      {labels.map((label, index) => {
        const color = label.background || "#000000";

        return [
          index ? ", " : null,
          <span key={label.id} style={{ color }}>
            {label.value}
          </span>,
        ];
      })}
    </Block>
  );
});

/** The labels group that drew this region (PolygonLabels for a polygon, BrushLabels for a brush stroke, ...). */
const regionLabelControl = (region: any) =>
  region.results?.find((result: any) => result.type?.endsWith("labels"))?.from_name;

/**
 * Change the region's label from the details panel. A click goes through Label.onLabelInteract, the label bar's
 * own path, so read-only, maxUsages, single choice and undo behave the same. Only the region's own group is offered:
 * the label bar ignores a label of another group with the same name (Brush "TE" on a polygon does nothing).
 */
export const RegionLabelPicker: FC<{ region: any }> = observer(({ region }) => {
  const labels = (regionLabelControl(region)?.tiedChildren ?? []).filter(
    (label: any) => label.visible !== false && !label.isEmpty,
  );

  if (!labels.length) return null;
  const current = new Set(
    region.labelings.flatMap((result: any) => (result.selectedLabels || []).map((label: any) => label.id)),
  );

  return (
    <Block name="label-picker" role="group" aria-label="Region label">
      {labels.map((label: any) => (
        <Elem
          key={label.id}
          name="option"
          tag="button"
          type="button"
          mod={{ current: current.has(label.id) }}
          aria-pressed={current.has(label.id)}
          style={{
            borderColor: label.background,
            background: current.has(label.id) ? chroma(label.background).alpha(0.35).css() : undefined,
          }}
          onClick={() => label.onLabelInteract()}
        >
          {label.value}
        </Elem>
      ))}
    </Block>
  );
});
