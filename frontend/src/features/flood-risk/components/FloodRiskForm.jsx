import { useTranslation } from "../../../i18n/useTranslation";
import { FIELD_GROUPS } from "../utils/fields";

const inputClass = (invalid) =>
  `mt-1.5 block w-full rounded-md border bg-surface px-3 py-2 text-base text-ink ${
    invalid ? "border-risk-high" : "border-rule"
  }`;

function NumberField({ name, feature, value, error, onChange }) {
  const { t, formatNumber } = useTranslation();
  const id = `field-${name}`;
  const hintId = `${id}-hint`;
  return (
    <div>
      <label htmlFor={id} className="block text-sm font-semibold">
        {t(`features.${name}`)}{" "}
        <span className="font-normal text-muted">({feature.unit})</span>
      </label>
      <input
        id={id}
        name={name}
        type="text"
        inputMode="decimal"
        autoComplete="off"
        value={value}
        onChange={(event) => onChange(name, event.target.value)}
        aria-invalid={Boolean(error)}
        aria-describedby={hintId}
        className={inputClass(error)}
      />
      <p
        id={hintId}
        className={`mt-1 text-sm ${error ? "text-risk-high" : "text-muted"}`}
      >
        {error
          ? t(error)
          : t("form.trainingRange", {
              min: formatNumber(feature.training_min),
              max: formatNumber(feature.training_max),
            })}
      </p>
    </div>
  );
}

export default function FloodRiskForm({
  metadata,
  values,
  errors,
  loading,
  onChange,
  onSubmit,
  onReset,
}) {
  const { t } = useTranslation();
  const features = Object.fromEntries(
    metadata.numeric_features.map((f) => [f.name, f]),
  );

  return (
    <form onSubmit={onSubmit} noValidate className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold">{t("form.title")}</h2>
        <p className="mt-2 max-w-prose text-muted">{t("form.intro")}</p>
      </div>

      <div>
        <label
          htmlFor="field-municipality"
          className="block text-sm font-semibold"
        >
          {t("form.municipality")}
        </label>
        <input
          id="field-municipality"
          type="text"
          maxLength={100}
          value={values.municipality}
          onChange={(event) => onChange("municipality", event.target.value)}
          aria-describedby="field-municipality-hint"
          className={inputClass(false)}
        />
        <p id="field-municipality-hint" className="mt-1 text-sm text-muted">
          {t("form.municipalityHint")}
        </p>
      </div>

      {FIELD_GROUPS.map((group) => (
        <fieldset key={group.key} className="border-t border-rule pt-5">
          <legend className="pr-3 text-lg font-bold">
            {t(`form.groups.${group.key}`)}
          </legend>
          <div className="mt-3 grid gap-5 sm:grid-cols-2">
            {group.fields.map((name) =>
              name === "soil_type" ? (
                <div key={name}>
                  <label
                    htmlFor="field-soil_type"
                    className="block text-sm font-semibold"
                  >
                    {t("features.soil_type")}
                  </label>
                  <select
                    id="field-soil_type"
                    value={values.soil_type}
                    onChange={(event) =>
                      onChange("soil_type", event.target.value)
                    }
                    aria-invalid={Boolean(errors.soil_type)}
                    className={inputClass(errors.soil_type)}
                  >
                    <option value="">{t("form.soilPlaceholder")}</option>
                    {metadata.soil_types.map((soil) => (
                      <option key={soil} value={soil}>
                        {soil}
                      </option>
                    ))}
                  </select>
                  {errors.soil_type && (
                    <p className="mt-1 text-sm text-risk-high">
                      {t(errors.soil_type)}
                    </p>
                  )}
                </div>
              ) : (
                <NumberField
                  key={name}
                  name={name}
                  feature={features[name]}
                  value={values[name]}
                  error={errors[name]}
                  onChange={onChange}
                />
              ),
            )}
          </div>
        </fieldset>
      ))}

      <div className="flex flex-wrap items-center gap-4 border-t border-rule pt-6">
        <button
          type="submit"
          disabled={loading}
          className="rounded-md bg-river px-6 py-3 text-base font-semibold text-white hover:bg-river-dark disabled:cursor-wait disabled:opacity-70"
        >
          {loading ? t("form.submitting") : t("form.submit")}
        </button>
        <button
          type="button"
          onClick={onReset}
          className="text-sm font-medium text-river underline underline-offset-4"
        >
          {t("form.reset")}
        </button>
      </div>
    </form>
  );
}
