// Thứ tự ưu tiên các control vẽ được đọc động từ Labeling Interface của project.
// Polygon đứng đầu để một nhãn dùng chung nhiều loại hình chỉ hiển thị một lần và ưu tiên công cụ Polygon.
const OBJECT_CONTROL_PRIORITY = [
  "polygonlabels",
  "rectanglelabels",
  "ellipsebasedpolygonlabels",
  "ellipselabels",
  "keypointlabels",
  "brushlabels",
];

const OBJECT_CONTROL_TYPES = new Set(OBJECT_CONTROL_PRIORITY);

// Chuẩn hóa giá trị kết quả thành chuỗi để so sánh được cả giá trị đơn và giá trị dạng object/array.
export const resultValueKey = (value) => JSON.stringify(value);

export const sameResultValue = (left, right) => resultValueKey(left) === resultValueKey(right);

// Giá trị lưu vào annotation ưu tiên resultValue/alias; giá trị hiển thị ưu tiên nội dung nhãn người dùng nhìn thấy.
export const childResultValue = (child) => child?.resultValue ?? child?.alias ?? child?._value ?? child?.value;

export const childDisplayValue = (child) => child?._value || child?.value || child?.alias || "";

// Tìm Choice/Label con theo đúng giá trị được lưu trong kết quả annotation.
export const findChildByResultValue = (control, value) =>
  control?.tiedChildren?.find((child) => sameResultValue(childResultValue(child), value));

// Chỉ kích hoạt nhãn trên control sẽ tạo region. Nếu cùng nhãn còn được chọn trên Rectangle,
// Ellipse hoặc Keypoint thì OpenCV có thể gắn tất cả kết quả đó vào một Polygon region.
export const activateObjectLabel = (controls, sourceControl, value) => {
  controls.forEach((control) => {
    control.unselectAll();
    if (control === sourceControl) findChildByResultValue(control, value)?.setSelected(true);
  });
};

// Lấy các control tạo vùng của frame hiện tại và sắp xếp theo thứ tự công cụ ưu tiên ở trên.
export const getObjectControls = (item) =>
  (item?.states?.() || [])
    .filter((control) => OBJECT_CONTROL_TYPES.has(control.type))
    .sort((left, right) => OBJECT_CONTROL_PRIORITY.indexOf(left.type) - OBJECT_CONTROL_PRIORITY.indexOf(right.type));

export const getObjectLabels = (item) => {
  const controls = getObjectControls(item);
  const seen = new Set();

  // Gộp nhãn từ mọi loại hình, bỏ nhãn ẩn và loại bản sao khi cùng giá trị xuất hiện ở nhiều control.
  return controls.flatMap((control) =>
    (control.tiedChildren || []).flatMap((label) => {
      const value = childResultValue(label);
      const key = resultValueKey(value);

      if (label.visible === false || seen.has(key)) return [];
      seen.add(key);
      return [{ control, key, label, value }];
    }),
  );
};

export const getEventControl = (item, frameIndex) => {
  const controls = (item?.states?.() || []).filter((control) => control.type === "choices");

  // Ưu tiên quy ước events-N của project hiện có, sau đó mới nhận tên Event tùy chỉnh hoặc Choices đầu tiên.
  return (
    controls.find((control) => control.name === `events-${frameIndex}`) ||
    controls.find((control) => /^events?[-_]/i.test(control.name)) ||
    controls[0]
  );
};

const clinicalControlName = (control) => control?.name?.toLowerCase?.() || "";

// Lấy các trường Clinical chuẩn từ Choices/TextArea có tên clinical-* và bỏ control JSON cũ clinical-manual.
export const getClinicalControls = (currentEntity) =>
  Array.from(currentEntity?.names?.values?.() || []).filter((control) => {
    const name = clinicalControlName(control);

    return name.startsWith("clinical-") && name !== "clinical-manual" && ["choices", "textarea"].includes(control.type);
  });

// Tách nhóm trường mở rộng để UI có thể thu gọn riêng khỏi các trường Clinical chính.
export const isExtendedClinicalControl = (control) => clinicalControlName(control).startsWith("clinical-extended-");

// Chuyển tên kỹ thuật như clinical-et-decision thành tiêu đề dễ đọc như ET Decision.
const humanizeControlName = (name) =>
  name
    .replace(/^clinical-(extended-)?/i, "")
    .split(/[-_]+/)
    .filter(Boolean)
    .map((word) => (word.length <= 3 ? word.toUpperCase() : `${word[0].toUpperCase()}${word.slice(1)}`))
    .join(" ");

export const getClinicalControlTitle = (control) => {
  const header = control?.children?.find((child) => child.type === "header");

  // Ưu tiên label/Header do project cấu hình; chỉ suy ra tiêu đề từ name khi không có hai giá trị trên.
  return control?.label || header?._value || header?.value || humanizeControlName(control?.name || "");
};
