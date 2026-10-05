import {
  activateObjectLabel,
  childResultValue,
  findChildByResultValue,
  getClinicalControls,
  getClinicalControlTitle,
  getEventControl,
  getObjectLabels,
  isExtendedClinicalControl,
} from "../EmbryoConfig";

describe("EmbryoPanel config discovery", () => {
  it("uses every custom label from supported image controls", () => {
    const item = {
      states: () => [
        { type: "keypointlabels", tiedChildren: [{ value: "Center", visible: true }] },
        {
          type: "polygonlabels",
          tiedChildren: [
            { value: "Custom object", visible: true },
            { value: "Hidden object", visible: false },
          ],
        },
      ],
    };

    expect(getObjectLabels(item).map(({ label }) => label.value)).toEqual([
      "Custom object",
      "Hidden object",
      "Center",
    ]);
  });

  it("deduplicates matching object labels and prefers polygon controls", () => {
    const item = {
      states: () => [
        { type: "rectanglelabels", name: "rect", tiedChildren: [{ value: "PN" }] },
        { type: "polygonlabels", name: "polygon", tiedChildren: [{ value: "PN" }] },
      ],
    };

    expect(getObjectLabels(item)).toMatchObject([{ control: { name: "polygon" }, value: "PN" }]);
  });

  it("prefers a visible lower-priority occurrence over a hidden one", () => {
    const item = {
      states: () => [
        { type: "polygonlabels", name: "polygon", tiedChildren: [{ value: "Membrane", visible: false }] },
        { type: "ellipselabels", name: "ellipse", tiedChildren: [{ value: "Membrane", visible: true }] },
      ],
    };

    expect(getObjectLabels(item)).toMatchObject([{ control: { name: "ellipse" }, value: "Membrane" }]);
  });

  it("keeps PN and PN Borderline as distinct labels without exact duplicates", () => {
    const item = {
      states: () => [
        {
          type: "polygonlabels",
          tiedChildren: [
            { value: "PN", visible: true },
            { value: "PN Borderline", visible: false },
          ],
        },
        { type: "rectanglelabels", tiedChildren: [{ value: "PN", visible: true }] },
      ],
    };

    expect(getObjectLabels(item).map(({ value }) => value)).toEqual(["PN", "PN Borderline"]);
  });

  it("activates an object label only on the control that creates the region", () => {
    const polygonLabel = { setSelected: jest.fn() };
    const polygon = { tiedChildren: [{ ...polygonLabel, value: "PN" }], unselectAll: jest.fn() };
    const rectangleLabel = { setSelected: jest.fn(), value: "PN" };
    const rectangle = { tiedChildren: [rectangleLabel], unselectAll: jest.fn() };

    activateObjectLabel([polygon, rectangle], polygon, "PN");

    expect(polygon.unselectAll).toHaveBeenCalledTimes(1);
    expect(rectangle.unselectAll).toHaveBeenCalledTimes(1);
    expect(polygon.tiedChildren[0].setSelected).toHaveBeenCalledWith(true);
    expect(rectangleLabel.setSelected).not.toHaveBeenCalled();
  });

  it("discovers custom event controls attached to a frame", () => {
    const generic = { type: "choices", name: "stage-events-4" };
    const exact = { type: "choices", name: "events-4" };
    const item = { states: () => [generic, exact] };

    expect(getEventControl(item, 4)).toBe(exact);
    expect(getEventControl({ states: () => [generic] }, 4)).toBe(generic);
  });

  it("uses result aliases when matching custom choices", () => {
    const choice = { alias: "stored-value", value: "Displayed value" };
    const control = { tiedChildren: [choice] };

    expect(childResultValue(choice)).toBe("stored-value");
    expect(findChildByResultValue(control, "stored-value")).toBe(choice);
  });

  it("discovers and labels clinical controls by naming convention", () => {
    const controls = [
      { type: "choices", name: "clinical-pn", children: [{ type: "header", value: "PN count" }] },
      { type: "textarea", name: "clinical-extended-notes", label: "Notes" },
      { type: "textarea", name: "clinical-manual" },
      { type: "choices", name: "pn_visibility" },
    ];
    const currentEntity = { names: new Map(controls.map((control) => [control.name, control])) };
    const clinical = getClinicalControls(currentEntity);

    expect(clinical).toEqual(controls.slice(0, 2));
    expect(getClinicalControlTitle(clinical[0])).toBe("PN count");
    expect(getClinicalControlTitle(clinical[1])).toBe("Notes");
    expect(isExtendedClinicalControl(clinical[1])).toBe(true);
  });
});
