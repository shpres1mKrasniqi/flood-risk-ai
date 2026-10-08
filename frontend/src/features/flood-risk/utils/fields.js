export const FIELD_GROUPS = [
  {
    key: "terrain",
    fields: ["elevation", "min_slope", "max_slope", "soil_type"],
  },
  {
    key: "water",
    fields: [
      "distance_from_river",
      "rainfall",
      "max_water_level",
      "min_water_level",
    ],
  },
];

export const MALISHEVA_DEFAULTS = {
  municipality: "Malishevë",
  elevation: "544",
  distance_from_river: "0.21",
  rainfall: "707.3",
  soil_type: "Smonice",
  max_water_level: "137",
  min_water_level: "41",
  min_slope: "2",
  max_slope: "16",
};

const PAIRS = [
  ["min_water_level", "max_water_level"],
  ["min_slope", "max_slope"],
];

export function validate(values, numericFeatures) {
  const errors = {};
  for (const name of numericFeatures) {
    const raw = String(values[name] ?? "")
      .trim()
      .replace(",", ".");
    if (raw === "") errors[name] = "validation.required";
    else if (Number.isNaN(Number(raw))) errors[name] = "validation.notNumber";
    else if (Number(raw) < 0) errors[name] = "validation.negative";
  }
  if (!values.soil_type) errors.soil_type = "validation.required";
  for (const [min, max] of PAIRS) {
    if (
      !errors[min] &&
      !errors[max] &&
      toNumber(values[min]) > toNumber(values[max])
    ) {
      errors[min] = "validation.minGreaterThanMax";
    }
  }
  return errors;
}

export const toNumber = (value) =>
  Number(String(value).trim().replace(",", "."));

export function toPayload(values, numericFeatures) {
  const payload = { soil_type: values.soil_type };
  for (const name of numericFeatures) payload[name] = toNumber(values[name]);
  const municipality = values.municipality?.trim();
  if (municipality) payload.municipality = municipality;
  return payload;
}

export const stripHeader = (text) =>
  text.replace(/^\s*Risk level:[^\n]*\n+/i, "").trim();
