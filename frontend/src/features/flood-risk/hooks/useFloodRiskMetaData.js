import { useEffect, useState } from "react";
import { getMetadata } from "../services/floodRiskApi";

export function useFloodRiskMetadata() {
  const [state, setState] = useState({
    metadata: null,
    error: null,
    loading: true,
  });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    getMetadata()
      .then(
        (metadata) =>
          active && setState({ metadata, error: null, loading: false }),
      )
      .catch(
        (error) =>
          active && setState({ metadata: null, error, loading: false }),
      );
    return () => {
      active = false;
    };
  }, [attempt]);

  const reload = () => {
    setState({ metadata: null, error: null, loading: true });
    setAttempt((n) => n + 1);
  };

  return { ...state, reload };
}
