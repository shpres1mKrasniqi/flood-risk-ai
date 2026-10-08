import { create } from "zustand";
import { persist } from "zustand/middleware";

export const SUPPORTED_LANGUAGES = ["sq", "en"];

export const useLanguageStore = create(
  persist(
    (set) => ({
      language: "sq",
      setLanguage: (language) => {
        if (SUPPORTED_LANGUAGES.includes(language)) set({ language });
      },
    }),
    { name: "flood-risk-language" },
  ),
);
