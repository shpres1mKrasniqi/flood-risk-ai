import {
  SUPPORTED_LANGUAGES,
  useLanguageStore,
} from "../../store/languageStore";
import { useTranslation } from "../../i18n/useTranslation";

export default function LanguageSwitcher() {
  const { t, language } = useTranslation();
  const setLanguage = useLanguageStore((state) => state.setLanguage);

  return (
    <div
      role="group"
      aria-label={t("language.label")}
      className="flex rounded-md border border-rule bg-surface p-0.5"
    >
      {SUPPORTED_LANGUAGES.map((code) => (
        <button
          key={code}
          type="button"
          lang={code}
          aria-pressed={language === code}
          onClick={() => setLanguage(code)}
          className={`rounded px-3 py-1.5 text-sm font-medium transition-colors ${
            language === code
              ? "bg-ink text-white"
              : "text-muted hover:text-ink"
          }`}
        >
          {t(`language.${code}`)}
        </button>
      ))}
    </div>
  );
}
