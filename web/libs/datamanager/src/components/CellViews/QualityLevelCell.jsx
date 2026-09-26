import React, { useState } from "react";
import { useSDK } from "../../providers/SDKProvider";
import { Button } from "../Common/Button/Button";

const LEVELS = {
  0: { label: "Not labeled", color: "#6b7280" },
  1: { label: "Annotator", color: "#2563eb" },
  2: { label: "Reviewed", color: "#d97706" },
  3: { label: "Confirmed", color: "#15803d" },
};

export const QualityLevelCell = ({ value, original }) => {
  const sdk = useSDK();
  const [changing, setChanging] = useState(false);
  const level = Math.min(Math.max(Number(value) || 0, 0), 3);
  const status = LEVELS[level];
  const changeLevel = async (event) => {
    event.stopPropagation();
    const annotationIDs = original?.quality_level_change_ids ?? [];
    if (!annotationIDs.length || changing || ![2, 3].includes(level)) return;

    const nextLevel = level === 2 ? 3 : 2;
    const taskID = original.task_id ?? original.id;
    const remainingIDs = [...annotationIDs];
    setChanging(true);
    try {
      while (remainingIDs.length) {
        const annotationID = remainingIDs[0];
        const response = await sdk.apiCall(
          "updateAnnotation",
          { taskID, annotationID },
          { body: { quality_level: nextLevel } },
        );
        const responseStatus = response?.$meta?.status;
        if (responseStatus !== 200 && responseStatus !== 201) {
          throw new Error("Unable to change annotation status");
        }
        remainingIDs.shift();
      }
      original.quality_level = nextLevel;
      original.quality_level_change_ids = annotationIDs;
      sdk.invoke("toast", {
        message: nextLevel === 3 ? "Task confirmed" : "Task returned to Reviewed",
        type: "info",
      });
    } catch (_error) {
      original.quality_level_change_ids = remainingIDs;
      sdk.invoke("toast", { message: "Unable to change task status", type: "error" });
    } finally {
      setChanging(false);
    }
  };

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <span
        title={`Labeling status: ${status.label}`}
        style={{ color: status.color, fontSize: 12, fontWeight: 500, whiteSpace: "nowrap" }}
      >
        {status.label}
      </span>
      {original?.quality_level_change_ids?.length && [2, 3].includes(level) ? (
        <Button disabled={changing} onClick={changeLevel} size="small" type="button">
          {changing ? "Updating..." : level === 2 ? "Confirm" : "Return to Reviewed"}
        </Button>
      ) : null}
    </div>
  );
};
