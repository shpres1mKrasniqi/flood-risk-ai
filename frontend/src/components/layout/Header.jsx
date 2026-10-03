import { useTranslation } from "../../i18n/useTranslation";
import LanguageSwitcher from "./LanguageSwitcher";

export default function Header() {
  const { t } = useTranslation();
  return (
    <header className="border-b border-rule">
      <div className="mx-auto flex max-w-6xl flex-wrap items-end justify-between gap-6 px-5 py-8 sm:px-8">
        <div className="max-w-2xl">
          <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl">
            {t("app.title")}
          </h1>
          <p className="mt-3 text-lg leading-relaxed text-muted">
            {t("app.subtitle")}
          </p>
        </div>
        <LanguageSwitcher />
      </div>
    </header>
  );
}
