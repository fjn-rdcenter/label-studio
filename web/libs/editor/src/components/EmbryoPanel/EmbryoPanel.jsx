import React, { useEffect, useState } from "react";
import { observer } from "mobx-react";

import { IconPolygonTool, IconRectangleTool } from "../../assets/icons";
import { getEmbryoNavigation, subscribeEmbryoNavigation } from "./EmbryoNavigation";
import styles from "./EmbryoPanel.module.scss";

const OBJECTS = [
  "PN",
  "MicroPN",
  "PB",
  "Blastomere (Cell)",
  "Fragment",
  "Vacuole",
  "Blastocoel",
  "ICM",
  "TE",
  "Hatching opening",
  "ZP",
  "Perivitelline space",
];

const EVENTS = [
  "t_pna",
  "t_pnf",
  "t_early cleavage",
  "t_fc",
  "t2",
  "t3",
  "t4",
  "t5",
  "t6",
  "t7",
  "t8",
  "t9+",
  "t_early compaction",
  "tM",
  "tSC",
  "t_early cavitation",
  "tSB",
  "tB",
  "tEB",
  "tHB",
];

const CLINICAL_FIELDS = [
  { name: "PN", options: ["0", "1", "2", "3", "4", "5", "Multi", "Deg"] },
  { name: "MicroPN", options: ["Yes", "No"] },
  { name: "Type of DC", options: ["DC1", "DC2", "Normal"] },
  { name: "Type of RC", options: ["RC1", "RC2", "Normal"] },
  { name: "Gardner" },
  { name: "ET decision", options: ["Discard", "cryopreservation", "embryo transfer"] },
];

const RARE_OBJECTS = new Set(["PB", "Vacuole"]);

const readClinical = (control) => {
  const value = control?.selectedValues?.()?.[0];

  if (!value) return {};
  try {
    return JSON.parse(value);
  } catch (_error) {
    return { Notes: value };
  }
};

const objectLabel = (region) => region.labels?.[0] || region.labelName || "Object";
const objectShape = (region) => region.type === "rectangleregion" ? "Rectangle" : "Polygon";
const objectShapeIcon = (region) => region.type === "rectangleregion" ? IconRectangleTool : IconPolygonTool;

const ObjectTab = observer(({ currentEntity, item }) => {
  const [showExtended, setShowExtended] = useState(false);
  const [checked, setChecked] = useState([]);
  const polygonControl = item.states?.()?.find(control => control.type === "polygonlabels");
  const rectangleControl = item.states?.()?.find(control => control.type === "rectanglelabels");
  const pnVisibilityControl = currentEntity.names.get("pn_visibility");
  const unknownVisibility = pnVisibilityControl?.findLabel("Unknown");
  const labels = polygonControl?.tiedChildren?.filter(label => OBJECTS.includes(label.value)) || [];
  const primaryLabels = labels.filter(label => !RARE_OBJECTS.has(label.value));
  const extendedLabels = labels.filter(label => RARE_OBJECTS.has(label.value));
  const regions = item.regs.filter(region => ["polygonregion", "rectangleregion"].includes(region.type));

  useEffect(() => {
    setChecked(current => current.filter(id => regions.some(region => region.id === id)));
  }, [item.currentImage, regions.length]);

  const selectLabel = (label) => {
    item.annotation.unselectAreas();
    [polygonControl, rectangleControl].forEach(control => {
      const matchingLabel = control?.findLabel(label.value);

      if (!matchingLabel) return;
      control.unselectAll();
      matchingLabel.setSelected(true);
    });
    const tool = polygonControl?.tools?.Polygon;
    if (tool) item.getToolsManager().selectTool(tool, true);
  };

  const renderLabel = label => {
    const selected = label.selected || rectangleControl?.findLabel(label.value)?.selected;

    return (
      <button
        className={`${styles.labelButton} ${selected ? styles.labelButtonSelected : ""}`}
        key={label.id}
        onClick={() => selectLabel(label)}
        type="button"
      >
        <span className={styles.swatch} style={{ background: label.background }} />
        <span>{label.value}</span>
        {label.hotkey ? <kbd>{label.hotkey}</kbd> : null}
      </button>
    );
  };

  const toggleChecked = id => {
    setChecked(current => current.includes(id) ? current.filter(value => value !== id) : [...current, id]);
  };

  const togglePnVisibility = () => {
    if (!pnVisibilityControl || currentEntity.isReadOnly()) return;
    const nextValues = unknownVisibility?.sel ? [] : ["Unknown"];

    pnVisibilityControl.setResult(nextValues);
    if (nextValues.length || pnVisibilityControl.result) pnVisibilityControl.updateResult();
  };

  const deleteChecked = () => {
    regions.filter(region => checked.includes(region.id)).forEach(region => region.deleteRegion());
    setChecked([]);
  };

  return (
    <>
      <div className={styles.labelGrid}>{primaryLabels.map(renderLabel)}</div>
      {extendedLabels.length ? (
        <div className={styles.extended}>
          <button className={styles.sectionButton} onClick={() => setShowExtended(value => !value)} type="button">
            Extended observation <span>{showExtended ? "-" : "+"}</span>
          </button>
          {showExtended ? <div className={styles.labelGrid}>{extendedLabels.map(renderLabel)}</div> : null}
        </div>
      ) : null}

      <div className={styles.regionHeading}>
        <strong>Select this checkbox if the Pronuclei cannot be seen</strong>
      </div>
      <label className={styles.eventRow}>
        <input checked={Boolean(unknownVisibility?.sel)} onChange={togglePnVisibility} type="checkbox" />
        <span>Unknown</span>
      </label>

      <div className={styles.regionHeading}>
        <strong>Objects on this slice</strong>
        <span>{regions.length}</span>
      </div>
      <div className={styles.regionList}>
        {regions.length === 0 ? <p className={styles.empty}>No objects on this slice.</p> : null}
        {regions.map((region, index) => {
          const ShapeIcon = objectShapeIcon(region);

          return (
            <div className={`${styles.regionRow} ${region.inSelection ? styles.regionSelected : ""}`} key={region.id}>
              <input
                aria-label={`Select ${objectLabel(region)}`}
                checked={checked.includes(region.id)}
                onChange={() => toggleChecked(region.id)}
                type="checkbox"
              />
              <button className={styles.regionName} onClick={() => item.annotation.selectArea(region)} type="button">
                <span className={styles.swatch} style={{ background: region.getOneColor?.() }} />
                <ShapeIcon className={styles.shapeIcon} title={objectShape(region)} />
                <span>{objectLabel(region)} {index + 1}</span>
              </button>
              <button className={styles.smallButton} onClick={() => region.toggleHidden()} type="button">
                {region.hidden ? "Show" : "Hide"}
              </button>
              <button className={styles.deleteButton} onClick={() => region.deleteRegion()} type="button">Delete</button>
            </div>
          );
        })}
      </div>
      {checked.length ? (
        <button className={styles.bulkDelete} onClick={deleteChecked} type="button">
          Delete selected ({checked.length})
        </button>
      ) : null}
    </>
  );
});

const EventTab = observer(({ currentEntity, frameIndex, frameData, paging }) => {
  const control = currentEntity.names.get(`events-${frameIndex}`);

  if (!control) return <p className={styles.empty}>Event controls are not configured for this project.</p>;

  const controls = frameData.map((_, index) => currentEntity.names.get(`events-${index}`));
  const updateChoice = (targetControl, value, selected) => {
    if (!targetControl || currentEntity.isReadOnly()) return;
    const selectedValues = targetControl.selectedValues();
    const nextValues = selected
      ? Array.from(new Set([...selectedValues, value]))
      : selectedValues.filter(selectedValue => selectedValue !== value);

    targetControl.setResult(nextValues);
    if (nextValues.length || targetControl.result) targetControl.updateResult();
  };

  const selectEvent = value => {
    controls.forEach((otherControl, index) => {
      if (index !== frameIndex && otherControl?.findLabel(value)?.sel) updateChoice(otherControl, value, false);
    });
    updateChoice(control, value, !control.findLabel(value)?.sel);
  };

  const navigateToEvent = selectedFrame => {
    paging.setFrame?.(selectedFrame);
    const targetImage = currentEntity.names.get(`timelapse_${selectedFrame}`);
    const defaultSlice = Math.min(5, Math.max((targetImage?.imageEntities?.length || 1) - 1, 0));

    targetImage?.setCurrentImage(defaultSlice);
  };

  return (
    <div className={styles.eventList}>
      {EVENTS.map(value => {
        const choice = control.findLabel(value);
        const selectedFrame = controls.findIndex(eventControl => eventControl?.findLabel(value)?.sel);
        const selectedTime = selectedFrame >= 0 ? frameData[selectedFrame]?.time : null;

        return (
          <div
            aria-checked={Boolean(choice?.sel)}
            className={styles.eventRow}
            key={value}
            onClick={() => selectEvent(value)}
            onKeyDown={event => {
              if (event.target === event.currentTarget && ["Enter", " "].includes(event.key)) {
                event.preventDefault();
                selectEvent(value);
              }
            }}
            role="checkbox"
            tabIndex={0}
          >
            <input
              checked={Boolean(choice?.sel)}
              onChange={() => selectEvent(value)}
              onClick={event => event.stopPropagation()}
              type="checkbox"
            />
            <span>{value}</span>
            {selectedFrame >= 0 ? (
              <button
                className={styles.eventTime}
                onClick={event => {
                  event.stopPropagation();
                  navigateToEvent(selectedFrame);
                }}
                onKeyDown={event => event.stopPropagation()}
                type="button"
              >
                {selectedTime || `Frame ${selectedFrame + 1}`}
              </button>
            ) : <span className={styles.eventTime}>Not set</span>}
          </div>
        );
      })}
    </div>
  );
});

const ClinicalTab = observer(({ currentEntity, taskData }) => {
  const control = currentEntity.names.get("clinical-manual");
  const imported = taskData.clinical || taskData.clinical_data || taskData.clinicalData;
  const [showImported, setShowImported] = useState(true);
  const [showExtended, setShowExtended] = useState(false);
  const [values, setValues] = useState(() => readClinical(control));

  useEffect(() => setValues(readClinical(control)), [control?.result?.id]);

  const save = (nextValues = values) => {
    if (!control || currentEntity.isReadOnly()) return;
    const serialized = JSON.stringify(nextValues);

    control.updateFromResult([serialized]);
    control.updateResult();
  };

  const setClinicalValue = (field, value, saveImmediately = false) => {
    const nextValues = { ...values, [field]: value };

    setValues(nextValues);
    if (saveImmediately) save(nextValues);
  };

  const renderClinicalField = ({ name, options }) => (
    <label key={name}>
      <span>{name}</span>
      {options ? (
        <select onChange={event => setClinicalValue(name, event.target.value, true)} value={values[name] || ""}>
          <option value="">Select...</option>
          {options.map(option => <option key={option} value={option}>{option}</option>)}
        </select>
      ) : (
        <input
          onBlur={() => save()}
          onChange={event => setClinicalValue(name, event.target.value)}
          value={values[name] || ""}
        />
      )}
    </label>
  );

  return (
    <>
      <button className={styles.sectionButton} onClick={() => setShowImported(value => !value)} type="button">
        Excel clinical data <span>{showImported ? "-" : "+"}</span>
      </button>
      {showImported ? (
        imported ? <pre className={styles.imported}>{JSON.stringify(imported, null, 2)}</pre> :
          <p className={styles.empty}>No imported clinical data in this task.</p>
      ) : null}
      <div className={styles.clinicalForm}>{CLINICAL_FIELDS.map(renderClinicalField)}</div>
      <div className={styles.extended}>
        <button className={styles.sectionButton} onClick={() => setShowExtended(value => !value)} type="button">
          Extended clinical <span>{showExtended ? "-" : "+"}</span>
        </button>
        {showExtended ? <div className={styles.clinicalForm}>{renderClinicalField({ name: "PB" })}</div> : null}
      </div>
    </>
  );
});

export const EmbryoPanel = observer(({ currentEntity }) => {
  const [paging, setPaging] = useState(getEmbryoNavigation);
  const [tab, setTab] = useState("object");
  const day = paging.day ?? 0;
  const frameIndex = Math.max((paging.page || 1) - 1, 0);
  const taskData = currentEntity.store?.task?.dataObj || {};
  const frameData = taskData.timelapse || [];
  const item = currentEntity.names.get(`timelapse_${frameIndex}`);

  useEffect(() => subscribeEmbryoNavigation(setPaging), []);

  if (!item) return <p className={styles.empty}>Open a timelapse frame to start annotation.</p>;

  return (
    <div className={styles.content}>
      <div className={styles.dayHeader}>
        <label htmlFor={`day-${item.id}`}>Development day</label>
        <select
          id={`day-${item.id}`}
          onChange={event => paging.setDay?.(Number(event.target.value))}
          value={day}
        >
          {[0, 1, 2, 3, 4, 5, 6].map(value => (
            <option disabled={!paging.availableDays.includes(value)} key={value} value={value}>
              DAY {value === 6 ? "6+" : value}
            </option>
          ))}
        </select>
        <span>{paging.frameLabel}</span>
      </div>
      <div className={styles.annotationCard}>
        <div className={styles.tabs} role="tablist">
          {["object", "event", "clinical"].map(value => (
            <button
              aria-selected={tab === value}
              className={tab === value ? styles.activeTab : ""}
              key={value}
              onClick={() => setTab(value)}
              role="tab"
              type="button"
            >
              {value[0].toUpperCase() + value.slice(1)}
            </button>
          ))}
        </div>
        <div className={styles.tabContent}>
          {tab === "object" ? <ObjectTab currentEntity={currentEntity} item={item} /> : null}
          {tab === "event" ? (
            <EventTab currentEntity={currentEntity} frameData={frameData} frameIndex={frameIndex} paging={paging} />
          ) : null}
          {tab === "clinical" ? (
            <ClinicalTab currentEntity={currentEntity} taskData={taskData} />
          ) : null}
        </div>
      </div>
    </div>
  );
});

const useEmbryoPanelData = (currentEntity) => {
  const [paging, setPaging] = useState(getEmbryoNavigation);
  const taskData = currentEntity.store?.task?.dataObj || {};
  const frameData = taskData.timelapse || [];
  const frameIndex = Math.max((paging.page || 1) - 1, 0);
  const item = currentEntity.names.get(`timelapse_${frameIndex}`);

  useEffect(() => subscribeEmbryoNavigation(setPaging), []);

  return { paging, taskData, frameData, frameIndex, item };
};

export const DevelopmentDayPanel = observer(({ currentEntity }) => {
  const { paging, item } = useEmbryoPanelData(currentEntity);
  const day = paging.day ?? 0;

  if (!item) return <p className={styles.empty}>Open a timelapse frame to select a development day.</p>;

  return (
    <div className={styles.content}>
      <div className={styles.dayHeader}>
        <label htmlFor={`sidebar-day-${item.id}`}>Development day</label>
        <select
          id={`sidebar-day-${item.id}`}
          onChange={event => paging.setDay?.(Number(event.target.value))}
          value={day}
        >
          {[0, 1, 2, 3, 4, 5, 6].map(value => (
            <option disabled={!paging.availableDays.includes(value)} key={value} value={value}>
              DAY {value === 6 ? "6+" : value}
            </option>
          ))}
        </select>
        <span>{paging.frameLabel}</span>
      </div>
    </div>
  );
});

export const ObjectPanel = observer(({ currentEntity }) => {
  const { item } = useEmbryoPanelData(currentEntity);

  return (
    <div className={styles.nativeTabContent}>
      {item ? (
        <ObjectTab currentEntity={currentEntity} item={item} />
      ) : (
        <p className={styles.empty}>Open a timelapse frame.</p>
      )}
    </div>
  );
});

export const EventPanel = observer(({ currentEntity }) => {
  const { paging, frameData, frameIndex } = useEmbryoPanelData(currentEntity);

  return (
    <div className={styles.nativeTabContent}>
      <EventTab currentEntity={currentEntity} frameData={frameData} frameIndex={frameIndex} paging={paging} />
    </div>
  );
});

export const ClinicalPanel = observer(({ currentEntity }) => {
  const { taskData } = useEmbryoPanelData(currentEntity);

  return (
    <div className={styles.nativeTabContent}>
      <ClinicalTab currentEntity={currentEntity} taskData={taskData} />
    </div>
  );
});
