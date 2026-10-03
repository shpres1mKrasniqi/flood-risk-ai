import { useTranslation } from "../../../i18n/useTranslation";

export default function LimitationsNote() {
  const { t } = useTranslation();
  return (
    <details className="border-t border-rule pt-5 text-sm">
      <summary className="cursor-pointer font-semibold">
        {t("limitations.title")}
      </summary>
      <ul className="mt-3 max-w-prose list-disc space-y-1.5 pl-5 text-muted">
        {t("limitations.items").map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </details>
  );
}
