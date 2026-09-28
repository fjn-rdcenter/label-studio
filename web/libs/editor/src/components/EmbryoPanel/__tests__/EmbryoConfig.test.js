import {
  childResultValue,
  findChildByResultValue,
  getClinicalControls,
  getClinicalControlTitle,
  getEventControl,
  getObjectLabels,
  isExtendedClinicalControl,
} from "../EmbryoConfig";

describe("EmbryoPanel config discovery", () => {
  it("uses custom visible labels from supported image controls", () => {
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

    expect(getObjectLabels(item).map(({ label }) => label.value)).toEqual(["Custom object", "Center"]);
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
