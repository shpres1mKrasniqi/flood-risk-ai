import { useEffect } from "react";
import { useLanguageStore } from "../store/languageStore";
import { translations } from "./translations";

const LOCALES = { sq: "sq-AL", en: "en-GB" };

export function useTranslation() {
  const language = useLanguageStore((state) => state.language);
  const dictionary = translations[language];

  useEffect(() => {
    document.documentElement.lang = language;
  }, [language]);

  const t = (key, params = {}) => {
    const value = key
      .split(".")
      .reduce((node, part) => node?.[part], dictionary);
    if (typeof value !== "string") return value ?? key;
    return value.replace(
      /\{(\w+)\}/g,
      (_, name) => params[name] ?? `{${name}}`,
    );
  };

  const formatNumber = (value, maximumFractionDigits = 2) =>
    new Intl.NumberFormat(LOCALES[language], {
      maximumFractionDigits,
      useGrouping: false,
    }).format(value);

  return { t, language, formatNumber };
}
