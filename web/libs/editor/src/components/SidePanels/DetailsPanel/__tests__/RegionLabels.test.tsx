/* global test, expect, describe, jest */
import Enzyme, { mount } from "enzyme";
import Adapter from "@wojtekmaj/enzyme-adapter-react-17";
import { RegionLabelPicker } from "../RegionLabels";

Enzyme.configure({ adapter: new Adapter() });

const label = (id: string, value: string, extra = {}) => ({
  id,
  value,
  background: "#FFC069",
  visible: true,
  isEmpty: false,
  onLabelInteract: jest.fn(),
  ...extra,
});

// a polygon labelled ZP; the Brush group has labels with the same names, which must not be offered
const polygonLabels = [
  label("p-icm", "ICM"),
  label("p-te", "TE"),
  label("p-zp", "ZP"),
  label("p-hidden", "X", { visible: false }),
];
const region = (labels = polygonLabels, selected = [polygonLabels[2]]) => ({
  results: [{ type: "polygonlabels", from_name: { tiedChildren: labels } }],
  labelings: [{ selectedLabels: selected }],
});

describe("RegionLabelPicker", () => {
  test("offers the visible labels of the region's own group and marks the current one", () => {
    const view = mount(<RegionLabelPicker region={region()} />);
    const buttons = view.find("button");

    expect(buttons.map((b) => b.text())).toEqual(["ICM", "TE", "ZP"]);
    expect(buttons.map((b) => b.prop("aria-pressed"))).toEqual([false, false, true]);
  });

  test("a click changes the label through the label bar's own path", () => {
    const view = mount(<RegionLabelPicker region={region()} />);

    view.find("button").at(1).simulate("click");
    expect(polygonLabels[1].onLabelInteract).toHaveBeenCalledTimes(1);
    expect(polygonLabels[2].onLabelInteract).not.toHaveBeenCalled();
  });

  test("an unlabeled region (e.g. merged from different labels) still gets the buttons", () => {
    const view = mount(<RegionLabelPicker region={region(polygonLabels, [])} />);

    expect(view.find("button")).toHaveLength(3);
    expect(view.find("button").map((b) => b.prop("aria-pressed"))).toEqual([false, false, false]);
  });

  test("nothing is rendered for a region without a labels group", () => {
    const view = mount(<RegionLabelPicker region={{ results: [{ type: "polygon" }], labelings: [] }} />);

    expect(view.find("button")).toHaveLength(0);
  });
});
