import { useTranslation } from "../../../i18n/useTranslation";
import InterpretationPanel from "./InterpretationPanel";
import RiskGauge from "./RiskGauge";

const MARKER = {
  Low: "bg-risk-low",
  Medium: "bg-risk-medium",
  High: "bg-risk-high",
};

export default function RiskResult({ result }) {
  const { t, formatNumber } = useTranslation();
  const heading = result.municipality
    ? t("result.riskFor", { municipality: result.municipality })
    : t("result.riskGeneric");

  return (
    <section aria-live="polite" className="space-y-8">
      <div>
        <h2 className="text-lg font-semibold text-muted">{heading}</h2>
        <p className="mt-1 flex items-center gap-3 text-5xl font-extrabold tracking-tight">
          <span
            className={`inline-block h-8 w-3 rounded-sm ${MARKER[result.risk]}`}
            aria-hidden="true"
          />
          {t(`risk.${result.risk}`)}
        </p>
      </div>

      <figure className="flex flex-col gap-6 sm:flex-row sm:items-end">
        <div className="w-60 shrink-0">
          <RiskGauge risk={result.risk} scores={result.class_scores} />
        </div>
        <figcaption className="max-w-xs text-sm text-muted">
          <span className="block font-semibold text-ink">
            {t("result.votes")}
          </span>
          {t("result.votesNote")}
        </figcaption>
      </figure>

      {result.out_of_range.length > 0 && (
        <div className="border-l-4 border-risk-medium bg-surface px-4 py-3">
          <h3 className="font-semibold">{t("result.outOfRangeTitle")}</h3>
          <ul className="mt-1 space-y-1 text-sm">
            {result.out_of_range.map((item) => (
              <li key={item.feature}>
                {t("result.outOfRange", {
                  feature: t(`features.${item.feature}`),
                  value: formatNumber(item.value),
                  min: formatNumber(item.training_min),
                  max: formatNumber(item.training_max),
                  unit: item.unit,
                })}
              </li>
            ))}
          </ul>
        </div>
      )}

      <InterpretationPanel
        status={result.interpretation_status}
        text={result.interpretation}
      />

      <p className="text-sm text-muted">
        {t("result.model", {
          mean: formatNumber(result.model.cv_macro_f1_mean),
          std: formatNumber(result.model.cv_macro_f1_std),
        })}
      </p>
    </section>
  );
}
