import React, { useEffect, useState } from "react";
import { observer } from "mobx-react";

import {
  IconBrushTool,
  IconCircleTool,
  IconKeypointsTool,
  IconPolygonTool,
  IconRectangleTool,
  IconRingPolygonTool,
} from "../../assets/icons";
import { getEmbryoNavigation, subscribeEmbryoNavigation } from "./EmbryoNavigation";
import {
  childDisplayValue,
  childResultValue,
  findChildByResultValue,
  getClinicalControls,
  getClinicalControlTitle,
  getEventControl,
  getObjectControls,
  getObjectLabels,
  isExtendedClinicalControl,
  sameResultValue,
} from "./EmbryoConfig";
import styles from "./EmbryoPanel.module.scss";

// Danh sách này chỉ dùng để đọc/ghi định dạng JSON clinical-manual của các project cũ.
const CLINICAL_FIELDS = [
  { name: "PN", options: ["0", "1", "2", "3", "4", "5", "Multi", "Deg"] },
  { name: "MicroPN", options: ["Yes", "No"] },
  { name: "Type of DC", options: ["DC1", "DC2", "Normal"] },
  { name: "Type of RC", options: ["RC1", "RC2", "Normal"] },
  { name: "Gardner" },
  { name: "ET decision", options: ["Discard", "cryopreservation", "embryo transfer"] },
];

// Các đối tượng ít dùng được đưa vào nhóm Extended observation để giao diện chính gọn hơn.
const RARE_OBJECTS = new Set(["PB", "Vacuole"]);

// Đọc dữ liệu Clinical JSON cũ; dữ liệu văn bản không phải JSON được giữ lại dưới trường Notes.
const readClinical = (control) => {
  const value = control?.selectedValues?.()?.[0];

  if (!value) return {};
  try {
    return JSON.parse(value);
  } catch (_error) {
    return { Notes: value };
  }
};

// Chuẩn hóa tên, loại hình và icon của mọi region để danh sách Object render thống nhất.
const objectLabel = (region) => region.labels?.[0] || region.labelName || "Object";
const objectShape = (region) =>
  ({
    brushregion: "Brush",
    ellipseregion: "Ellipse",
    keypointregion: "Keypoint",
    polygonregion: region.ring ? "Ring Polygon" : "Polygon",
    rectangleregion: "Rectangle",
  })[region.type] || "Object";
const objectShapeIcon = (region) =>
  ({
    brushregion: IconBrushTool,
    ellipseregion: IconCircleTool,
    keypointregion: IconKeypointsTool,
    polygonregion: region.ring ? IconRingPolygonTool : IconPolygonTool,
    rectangleregion: IconRectangleTool,
  })[region.type] || IconPolygonTool;

const setChoiceValue = (control, value, selected, readOnly) => {
  // Không ghi khi thiếu control hoặc annotation đang ở chế độ chỉ đọc.
  if (!control || readOnly) return;
  const selectedValues = control.selectedValues();
  const withoutValue = selectedValues.filter((selectedValue) => !sameResultValue(selectedValue, value));

  // Multiple giữ các lựa chọn khác; Single luôn thay bằng đúng lựa chọn mới.
  const nextValues = selected ? (control.choice === "multiple" ? [...withoutValue, value] : [value]) : withoutValue;

  // Đồng bộ kết quả vào MobX model và phát cập nhật để Label Studio lưu annotation.
  control.setResult(nextValues);
  if (nextValues.length || control.result) control.updateResult();
};

// Luồng Object: đọc nhãn đã cấu hình, đồng bộ nhãn trùng giữa các loại hình,
// rồi kích hoạt công cụ của control có độ ưu tiên cao nhất do EmbryoConfig chọn.
const ObjectTab = observer(({ currentEntity, item }) => {
  const [showExtended, setShowExtended] = useState(false);
  const [checked, setChecked] = useState([]);
  const objectControls = getObjectControls(item);
  const pnVisibilityControl = currentEntity.names.get("pn_visibility");
  const visibilityChoices = pnVisibilityControl?.tiedChildren?.filter((choice) => choice.visible !== false) || [];
  const visibilityLabel = currentEntity.names.get(pnVisibilityControl?.toname);
  const labels = getObjectLabels(item);
  const primaryLabels = labels.filter(({ label }) => !RARE_OBJECTS.has(childDisplayValue(label)));
  const extendedLabels = labels.filter(({ label }) => RARE_OBJECTS.has(childDisplayValue(label)));
  const regions = item.regs.filter((region) =>
    ["brushregion", "ellipseregion", "keypointregion", "polygonregion", "rectangleregion"].includes(region.type),
  );

  // Khi đổi ảnh/slice, bỏ các ID đã chọn nhưng không còn tồn tại trong danh sách region hiện tại.
  useEffect(() => {
    setChecked((current) => current.filter((id) => regions.some((region) => region.id === id)));
  }, [item.currentImage, regions.length]);

  const selectLabel = ({ control: sourceControl, value }) => {
    // Bỏ chọn region cũ rồi chọn cùng một nhãn trên mọi control hình học có hỗ trợ nhãn đó.
    item.annotation.unselectAreas();
    objectControls.forEach((control) => {
      const matchingLabel = findChildByResultValue(control, value);

      if (!matchingLabel) return;
      control.unselectAll();
      matchingLabel.setSelected(true);
    });

    // Giữ công cụ hiện tại nếu thuộc đúng control; nếu không thì chọn công cụ mặc định của control nguồn.
    const manager = item.getToolsManager();
    const selectedTool = manager.findSelectedTool();
    const tool =
      selectedTool?.control?.name === sourceControl.name ? selectedTool : Object.values(sourceControl.tools || {})[0];
    if (tool) manager.selectTool(tool, true);
  };

  const renderLabel = (descriptor) => {
    const { key, label, value } = descriptor;
    const selected = objectControls.some((control) => findChildByResultValue(control, value)?.selected);

    return (
      // Nút nhãn hiển thị màu, tên, hotkey và trạng thái đang được chọn trên bất kỳ control hình học nào.
      <button
        className={`${styles.labelButton} ${selected ? styles.labelButtonSelected : ""}`}
        key={key}
        onClick={() => selectLabel(descriptor)}
        type="button"
      >
        <span className={styles.swatch} style={{ background: label.background }} />
        <span>{childDisplayValue(label)}</span>
        {label.hotkey ? <kbd>{label.hotkey}</kbd> : null}
      </button>
    );
  };

  const toggleChecked = (id) => {
    // Quản lý danh sách region được đánh dấu để hỗ trợ xóa hàng loạt.
    setChecked((current) => (current.includes(id) ? current.filter((value) => value !== id) : [...current, id]));
  };

  const deleteChecked = () => {
    // Xóa các region đã đánh dấu khỏi annotation rồi làm sạch trạng thái checkbox.
    regions.filter((region) => checked.includes(region.id)).forEach((region) => region.deleteRegion());
    setChecked([]);
  };

  return (
    <>
      <div className={styles.labelGrid}>{primaryLabels.map(renderLabel)}</div>
      {labels.length === 0 ? (
        <p className={styles.empty}>
          Add image label controls in the project's Labeling Interface to configure objects.
        </p>
      ) : null}
      {extendedLabels.length ? (
        <div className={styles.extended}>
          <button className={styles.sectionButton} onClick={() => setShowExtended((value) => !value)} type="button">
            Extended observation <span>{showExtended ? "-" : "+"}</span>
          </button>
          {showExtended ? <div className={styles.labelGrid}>{extendedLabels.map(renderLabel)}</div> : null}
        </div>
      ) : null}

      {/* Render các lựa chọn về khả năng quan sát nếu project có cấu hình control pn_visibility. */}
      {visibilityChoices.length ? (
        <>
          <div className={styles.regionHeading}>
            <strong>{visibilityLabel?._value || visibilityLabel?.value || "Object visibility"}</strong>
          </div>
          {visibilityChoices.map((choice) => {
            const value = childResultValue(choice);

            return (
              <label className={styles.eventRow} key={JSON.stringify(value)}>
                <input
                  checked={Boolean(choice.sel)}
                  disabled={currentEntity.isReadOnly()}
                  onChange={() => setChoiceValue(pnVisibilityControl, value, !choice.sel, currentEntity.isReadOnly())}
                  type="checkbox"
                />
                <span>{childDisplayValue(choice)}</span>
              </label>
            );
          })}
        </>
      ) : null}

      {/* Danh sách region của slice hiện tại hỗ trợ chọn, ẩn/hiện, xóa đơn và xóa hàng loạt. */}
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
                <span>
                  {objectLabel(region)} {index + 1}
                </span>
              </button>
              <button className={styles.smallButton} onClick={() => region.toggleHidden()} type="button">
                {region.hidden ? "Show" : "Hide"}
              </button>
              <button className={styles.deleteButton} onClick={() => region.deleteRegion()} type="button">
                Delete
              </button>
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

// Luồng Event: Mỗi Event chỉ được thuộc một frame timelapse. Khi chọn Event tại đây, cùng Event ở frame khác
// sẽ bị bỏ chọn trước khi kết quả được lưu vào frame hiện tại.
const EventTab = observer(({ currentEntity, frameIndex, frameData, paging }) => {
  // Tìm Choices control Event của frame đang mở theo cấu hình động của project.
  const item = currentEntity.names.get(`timelapse_${frameIndex}`);
  const control = getEventControl(item, frameIndex);

  if (!control) {
    return (
      <p className={styles.empty}>Add a Choices control to each timelapse frame in the project's Labeling Interface.</p>
    );
  }

  const controls = frameData.map((_, index) => getEventControl(currentEntity.names.get(`timelapse_${index}`), index));
  const choices = control.tiedChildren?.filter((choice) => choice.visible !== false) || [];

  const selectEvent = (value) => {
    // Xóa Event trùng khỏi tất cả frame khác để bảo đảm tính duy nhất trên toàn timelapse.
    controls.forEach((otherControl, index) => {
      const otherChoice = findChildByResultValue(otherControl, value);

      if (index !== frameIndex && otherChoice?.sel) {
        setChoiceValue(otherControl, value, false, currentEntity.isReadOnly());
      }
    });

    // Đảo trạng thái Event trên frame hiện tại sau khi đã xử lý các frame còn lại.
    const choice = findChildByResultValue(control, value);

    setChoiceValue(control, value, !choice?.sel, currentEntity.isReadOnly());
  };

  const navigateToEvent = (selectedFrame) => {
    // Chuyển tới frame chứa Event và mở slice mặc định, tối đa là slice thứ sáu.
    paging.setFrame?.(selectedFrame);
    const targetImage = currentEntity.names.get(`timelapse_${selectedFrame}`);
    const defaultSlice = Math.min(5, Math.max((targetImage?.imageEntities?.length || 1) - 1, 0));

    targetImage?.setCurrentImage(defaultSlice);
  };

  return (
    <div className={styles.eventList}>
      {/* Hiển thị thời điểm đã gán; bấm thời điểm để quay nhanh tới frame chứa Event. */}
      {choices.map((choice) => {
        const value = childResultValue(choice);
        const selectedFrame = controls.findIndex((eventControl) => findChildByResultValue(eventControl, value)?.sel);
        const selectedTime = selectedFrame >= 0 ? frameData[selectedFrame]?.time : null;

        return (
          <div
            aria-checked={Boolean(choice?.sel)}
            className={styles.eventRow}
            key={JSON.stringify(value)}
            onClick={() => selectEvent(value)}
            onKeyDown={(event) => {
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
              onClick={(event) => event.stopPropagation()}
              type="checkbox"
            />
            <span>{childDisplayValue(choice)}</span>
            {selectedFrame >= 0 ? (
              <button
                className={styles.eventTime}
                onClick={(event) => {
                  event.stopPropagation();
                  navigateToEvent(selectedFrame);
                }}
                onKeyDown={(event) => event.stopPropagation()}
                type="button"
              >
                {selectedTime || `Frame ${selectedFrame + 1}`}
              </button>
            ) : (
              <span className={styles.eventTime}>Not set</span>
            )}
          </div>
        );
      })}
    </div>
  );
});

// Luồng Clinical: Project mới lưu từng trường Clinical thành kết quả Label Studio chuẩn thay vì gom vào một chuỗi JSON.
const ClinicalChoiceField = observer(({ control, currentEntity }) => {
  const choices = control.tiedChildren?.filter((choice) => choice.visible !== false) || [];
  const selectedValues = control.selectedValues();
  const title = getClinicalControlTitle(control);

  if (control.choice === "multiple") {
    // Choices loại multiple dùng checkbox để người dùng chọn đồng thời nhiều giá trị.
    return (
      <fieldset className={styles.clinicalChoices}>
        <legend>{title}</legend>
        {choices.map((choice) => {
          const value = childResultValue(choice);

          return (
            <label key={JSON.stringify(value)}>
              <input
                checked={selectedValues.some((selected) => sameResultValue(selected, value))}
                disabled={currentEntity.isReadOnly()}
                onChange={() => setChoiceValue(control, value, !choice.sel, currentEntity.isReadOnly())}
                type="checkbox"
              />
              <span>{childDisplayValue(choice)}</span>
            </label>
          );
        })}
      </fieldset>
    );
  }

  const selectedIndex = choices.findIndex((choice) =>
    selectedValues.some((selected) => sameResultValue(selected, childResultValue(choice))),
  );

  // Choices loại single dùng select; option rỗng có nhiệm vụ xóa kết quả hiện có.
  return (
    <label>
      <span>{title}</span>
      <select
        disabled={currentEntity.isReadOnly()}
        onChange={(event) => {
          if (event.target.value === "") {
            control.setResult([]);
            if (control.result) control.updateResult();
            return;
          }
          const choice = choices[Number(event.target.value)];

          setChoiceValue(control, childResultValue(choice), true, currentEntity.isReadOnly());
        }}
        value={selectedIndex < 0 ? "" : selectedIndex}
      >
        <option value="">Select...</option>
        {choices.map((choice, index) => (
          <option key={JSON.stringify(childResultValue(choice))} value={index}>
            {childDisplayValue(choice)}
          </option>
        ))}
      </select>
    </label>
  );
});

const ClinicalTextField = observer(({ control, currentEntity }) => {
  // Giữ bản nháp trong state và nạp lại khi người dùng chuyển sang một annotation/result khác.
  const [value, setValue] = useState(() => control.selectedValues?.()?.[0] || "");

  useEffect(() => setValue(control.selectedValues?.()?.[0] || ""), [control.result?.id]);

  const save = () => {
    // TextArea được ghi khi input mất focus để tránh cập nhật annotation sau từng phím bấm.
    if (currentEntity.isReadOnly()) return;
    control.updateFromResult(value ? [value] : []);
    if (value || control.result) control.updateResult();
  };

  return (
    <label>
      <span>{getClinicalControlTitle(control)}</span>
      <input
        disabled={currentEntity.isReadOnly()}
        onBlur={save}
        onChange={(event) => setValue(event.target.value)}
        placeholder={control.placeholder || ""}
        value={value}
      />
    </label>
  );
});

// Chọn component hiển thị theo đúng loại control đã khai báo trong Labeling Interface.
const ClinicalControl = ({ control, currentEntity }) =>
  control.type === "choices" ? (
    <ClinicalChoiceField control={control} currentEntity={currentEntity} />
  ) : (
    <ClinicalTextField control={control} currentEntity={currentEntity} />
  );

const ClinicalTab = observer(({ currentEntity, taskData }) => {
  // Ưu tiên các control clinical-* mới; clinical-manual chỉ còn là đường tương thích cho project cũ.
  const control = currentEntity.names.get("clinical-manual");
  const configuredControls = getClinicalControls(currentEntity);
  const primaryControls = configuredControls.filter(
    (configuredControl) => !isExtendedClinicalControl(configuredControl),
  );
  const extendedControls = configuredControls.filter(isExtendedClinicalControl);
  const imported = taskData.clinical || taskData.clinical_data || taskData.clinicalData;
  const [showImported, setShowImported] = useState(true);
  const [showExtended, setShowExtended] = useState(false);
  const [values, setValues] = useState(() => readClinical(control));

  // Nạp lại JSON cũ khi result hiện hành thay đổi.
  useEffect(() => setValues(readClinical(control)), [control?.result?.id]);

  const save = (nextValues = values) => {
    // Tuần tự hóa toàn bộ dữ liệu cũ thành một chuỗi JSON trước khi cập nhật result.
    if (!control || currentEntity.isReadOnly()) return;
    const serialized = JSON.stringify(nextValues);

    control.updateFromResult([serialized]);
    control.updateResult();
  };

  const setClinicalValue = (field, value, saveImmediately = false) => {
    // Select lưu ngay; input văn bản chỉ cập nhật state và sẽ lưu khi mất focus.
    const nextValues = { ...values, [field]: value };

    setValues(nextValues);
    if (saveImmediately) save(nextValues);
  };

  const renderClinicalField = ({ name, options }) => (
    <label key={name}>
      <span>{name}</span>
      {options ? (
        <select onChange={(event) => setClinicalValue(name, event.target.value, true)} value={values[name] || ""}>
          <option value="">Select...</option>
          {options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      ) : (
        <input
          onBlur={() => save()}
          onChange={(event) => setClinicalValue(name, event.target.value)}
          value={values[name] || ""}
        />
      )}
    </label>
  );

  return (
    <>
      <button className={styles.sectionButton} onClick={() => setShowImported((value) => !value)} type="button">
        Excel clinical data <span>{showImported ? "-" : "+"}</span>
      </button>
      {showImported ? (
        imported ? (
          <pre className={styles.imported}>{JSON.stringify(imported, null, 2)}</pre>
        ) : (
          <p className={styles.empty}>No imported clinical data in this task.</p>
        )
      ) : null}
      {/* Nếu có clinical-* thì render control chuẩn và tách nhóm extended thành khu vực thu gọn. */}
      {configuredControls.length ? (
        <>
          <div className={styles.clinicalForm}>
            {primaryControls.map((configuredControl) => (
              <ClinicalControl control={configuredControl} currentEntity={currentEntity} key={configuredControl.name} />
            ))}
          </div>
          {extendedControls.length ? (
            <div className={styles.extended}>
              <button className={styles.sectionButton} onClick={() => setShowExtended((value) => !value)} type="button">
                Extended clinical <span>{showExtended ? "-" : "+"}</span>
              </button>
              {showExtended ? (
                <div className={styles.clinicalForm}>
                  {extendedControls.map((configuredControl) => (
                    <ClinicalControl
                      control={configuredControl}
                      currentEntity={currentEntity}
                      key={configuredControl.name}
                    />
                  ))}
                </div>
              ) : null}
            </div>
          ) : null}
        </>
      ) : control ? (
        <>
          {/* Giữ định dạng JSON clinical-manual cho project được tạo trước khi có control clinical-*. */}
          <div className={styles.clinicalForm}>{CLINICAL_FIELDS.map(renderClinicalField)}</div>
          <div className={styles.extended}>
            <button className={styles.sectionButton} onClick={() => setShowExtended((value) => !value)} type="button">
              Extended clinical <span>{showExtended ? "-" : "+"}</span>
            </button>
            {showExtended ? <div className={styles.clinicalForm}>{renderClinicalField({ name: "PB" })}</div> : null}
          </div>
        </>
      ) : (
        <p className={styles.empty}>
          Add Choices or TextArea controls named clinical-* in the project's Labeling Interface.
        </p>
      )}
    </>
  );
});

export const EmbryoPanel = observer(({ currentEntity }) => {
  // Theo dõi navigation dùng chung để xác định ngày, frame, task data và ảnh timelapse hiện tại.
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
        <select id={`day-${item.id}`} onChange={(event) => paging.setDay?.(Number(event.target.value))} value={day}>
          {[0, 1, 2, 3, 4, 5, 6].map((value) => (
            <option disabled={!paging.availableDays.includes(value)} key={value} value={value}>
              DAY {value === 6 ? "6+" : value}
            </option>
          ))}
        </select>
        <span>{paging.frameLabel}</span>
      </div>
      <div className={styles.annotationCard}>
        {/* Panel gộp cũ giữ ba tab; các panel export bên dưới dùng khi sidebar render từng tab riêng. */}
        <div className={styles.tabs} role="tablist">
          {["object", "event", "clinical"].map((value) => (
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
          {tab === "clinical" ? <ClinicalTab currentEntity={currentEntity} taskData={taskData} /> : null}
        </div>
      </div>
    </div>
  );
});

const useEmbryoPanelData = (currentEntity) => {
  // Chuẩn hóa dữ liệu dùng chung để Development Day/Object/Event/Clinical không lặp logic tìm frame.
  const [paging, setPaging] = useState(getEmbryoNavigation);
  const taskData = currentEntity.store?.task?.dataObj || {};
  const frameData = taskData.timelapse || [];
  const frameIndex = Math.max((paging.page || 1) - 1, 0);
  const item = currentEntity.names.get(`timelapse_${frameIndex}`);

  useEffect(() => subscribeEmbryoNavigation(setPaging), []);

  return { paging, taskData, frameData, frameIndex, item };
};

export const DevelopmentDayPanel = observer(({ currentEntity }) => {
  // Panel nhỏ chỉ điều khiển ngày phát triển và hiển thị nhãn frame hiện tại.
  const { paging, item } = useEmbryoPanelData(currentEntity);
  const day = paging.day ?? 0;

  if (!item) return <p className={styles.empty}>Open a timelapse frame to select a development day.</p>;

  return (
    <div className={styles.content}>
      <div className={styles.dayHeader}>
        <label htmlFor={`sidebar-day-${item.id}`}>Development day</label>
        <select
          id={`sidebar-day-${item.id}`}
          onChange={(event) => paging.setDay?.(Number(event.target.value))}
          value={day}
        >
          {[0, 1, 2, 3, 4, 5, 6].map((value) => (
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
  // Adapter để sidebar native render riêng nội dung ObjectTab.
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
  // Adapter truyền dữ liệu frame và navigation vào EventTab của sidebar native.
  const { paging, frameData, frameIndex } = useEmbryoPanelData(currentEntity);

  return (
    <div className={styles.nativeTabContent}>
      <EventTab currentEntity={currentEntity} frameData={frameData} frameIndex={frameIndex} paging={paging} />
    </div>
  );
});

export const ClinicalPanel = observer(({ currentEntity }) => {
  // Adapter truyền task data vào ClinicalTab của sidebar native.
  const { taskData } = useEmbryoPanelData(currentEntity);

  return (
    <div className={styles.nativeTabContent}>
      <ClinicalTab currentEntity={currentEntity} taskData={taskData} />
    </div>
  );
});
