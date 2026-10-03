import { useTranslation } from "../../../i18n/useTranslation";
import { stripHeader } from "../utils/fields";

export default function InterpretationPanel({ status, text }) {
  const { t } = useTranslation();
  return (
    <section className="border-t border-rule pt-6">
      <h3 className="text-xl font-bold">{t("interpretation.title")}</h3>
      {status === "ok" && text ? (
        <>
          <div className="mt-3 max-w-prose space-y-3 text-base leading-relaxed">
            {stripHeader(text)
              .split(/\n\s*\n/)
              .map((paragraph, i) => (
                <p key={i}>{paragraph}</p>
              ))}
          </div>
          <p className="mt-4 max-w-prose text-sm text-muted">
            {t("interpretation.note")}
          </p>
        </>
      ) : (
        <p className="mt-3 max-w-prose text-muted">
          {t(`interpretation.${status}`)}
        </p>
      )}
    </section>
  );
}
