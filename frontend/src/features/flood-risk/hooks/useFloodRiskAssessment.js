import { useState } from "react";
import { explainFloodRisk } from "../services/floodRiskApi";

export function useFloodRiskAssessment() {
  const [state, setState] = useState({
    result: null,
    error: null,
    loading: false,
  });

  const assess = async (payload, language) => {
    setState((previous) => ({ ...previous, error: null, loading: true }));
    try {
      const result = await explainFloodRisk(payload, language);
      setState({ result, error: null, loading: false });
    } catch (error) {
      setState({ result: null, error, loading: false });
    }
  };

  return { ...state, assess };
}
