import { useState } from "react";
import { useTranslation } from "../i18n/useTranslation";
import FloodRiskForm from "../features/flood-risk/components/FloodRiskForm";
import LimitationsNote from "../features/flood-risk/components/LimitationsNote";
import RiskResult from "../features/flood-risk/components/RiskResult";
import { useFloodRiskAssessment } from "../features/flood-risk/hooks/useFloodRiskAssessment";
import { useFloodRiskMetadata } from "../features/flood-risk/hooks/useFloodRiskMetaData";
import {
  MALISHEVA_DEFAULTS,
  toPayload,
  validate,
} from "../features/flood-risk/utils/fields";

function ErrorMessage({ error, onRetry }) {
  const { t } = useTranslation();
  const message =
    error.kind === "unreachable"
      ? t("errors.unreachable", { url: error.detail })
      : t("errors.server", { detail: error.detail });
  return (
    <div
      role="alert"
      className="border-l-4 border-risk-high bg-surface px-4 py-3"
    >
      <p>{message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-2 font-semibold text-river underline underline-offset-4"
        >
          {t("errors.retry")}
        </button>
      )}
    </div>
  );
}

export default function FloodRiskPage() {
  const { t, language } = useTranslation();
  const {
    metadata,
    error: metadataError,
    loading: metadataLoading,
    reload,
  } = useFloodRiskMetadata();
  const { result, error, loading, assess } = useFloodRiskAssessment();
  const [values, setValues] = useState(MALISHEVA_DEFAULTS);
  const [errors, setErrors] = useState({});

  if (metadataLoading) return <p className="text-muted">…</p>;
  if (metadataError)
    return <ErrorMessage error={metadataError} onRetry={reload} />;

  const numericFeatures = metadata.numeric_features.map((f) => f.name);

  const handleChange = (name, value) => {
    setValues((previous) => ({ ...previous, [name]: value }));
    setErrors((previous) => ({ ...previous, [name]: undefined }));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    const found = validate(values, numericFeatures);
    setErrors(found);
    if (Object.keys(found).length === 0)
      assess(toPayload(values, numericFeatures), language);
  };

  const handleReset = () => {
    setValues(MALISHEVA_DEFAULTS);
    setErrors({});
  };

  return (
    <div className="grid gap-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:gap-16">
      <FloodRiskForm
        metadata={metadata}
        values={values}
        errors={errors}
        loading={loading}
        onChange={handleChange}
        onSubmit={handleSubmit}
        onReset={handleReset}
      />

      <div className="space-y-8 lg:border-l lg:border-rule lg:pl-16">
        {Object.values(errors).some(Boolean) && (
          <p role="alert" className="text-risk-high">
            {t("validation.fixFields")}
          </p>
        )}
        {error && <ErrorMessage error={error} />}
        {result ? (
          <RiskResult result={result} />
        ) : (
          !error && (
            <div>
              <h2 className="text-2xl font-bold">{t("result.emptyTitle")}</h2>
              <p className="mt-2 max-w-prose text-muted">{t("result.empty")}</p>
            </div>
          )
        )}
        <LimitationsNote />
      </div>
    </div>
  );
}
