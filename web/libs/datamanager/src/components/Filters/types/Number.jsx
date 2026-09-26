import { observer } from "mobx-react";
import React from "react";
import { isDefined } from "../../../utils/utils";
import { FilterInput } from "../FilterInput";
import { VariantSelect } from "./List";

const valueFilter = (value) => {
  if (isDefined(value)) {
    if (typeof value === "number") {
      return value;
    }
    if (typeof value === "string") {
      return value.replace(/([^\d.,]+)/, "");
    }
    return value || null;
  }

  return null;
};

const NumberInput = observer(({ onChange, ...rest }) => {
  return <FilterInput {...rest} type="number" onChange={(value) => onChange(valueFilter(value))} />;
});

const NumberInputWithSchema = (props) => {
  return props.schema?.items ? <VariantSelect {...props} /> : <NumberInput {...props} />;
};

const RangeInput = observer(({ schema, value, onChange }) => {
  const min = value?.min ?? null;
  const max = value?.max ?? null;

  const onValueChange = (newValue) => {
    console.log({ newValue });
    onChange(newValue);
  };

  const onChangeMin = (newValue) => {
    onValueChange({ min: Number(newValue), max });
  };

  const onChangeMax = (newValue) => {
    onValueChange({ min, max: Number(newValue) });
  };

  return (
    <>
      <NumberInput placeholder="Min" value={min} onChange={onChangeMin} schema={schema} style={{ flex: 1 }} />
      <span style={{ padding: "0 10px" }}>and</span>
      <NumberInput placeholder="Max" value={max} onChange={onChangeMax} schema={schema} style={{ flex: 1 }} />
    </>
  );
});

export const NumberFilter = [
  {
    key: "equal",
    label: "=",
    valueType: "single",
    input: (props) => <NumberInputWithSchema {...props} />,
  },
  {
    key: "not_equal",
    label: "≠",
    valueType: "single",
    input: (props) => <NumberInputWithSchema {...props} />,
  },
  {
    key: "less",
    label: "<",
    valueType: "single",
    input: (props) => <NumberInputWithSchema {...props} />,
  },
  {
    key: "greater",
    label: ">",
    valueType: "single",
    input: (props) => <NumberInputWithSchema {...props} />,
  },
  {
    key: "less_or_equal",
    label: "≤",
    valueType: "single",
    input: (props) => <NumberInputWithSchema {...props} />,
  },
  {
    key: "greater_or_equal",
    label: "≥",
    valueType: "single",
    input: (props) => <NumberInputWithSchema {...props} />,
  },
  {
    key: "in",
    label: "is between",
    valueType: "range",
    input: (props) => <RangeInput {...props} />,
  },
  {
    key: "not_in",
    label: "not between",
    valueType: "range",
    input: (props) => <RangeInput {...props} />,
  },
];
