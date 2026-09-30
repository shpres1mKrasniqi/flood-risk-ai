import json

import openai

from app.domain.flood_risk import (
    ClassifierMetadata,
    FloodRiskAssessment,
    InterpretationUnavailable,
    RiskInterpreter,
)

INSTRUCTIONS = """\
You explain the output of a machine-learning flood-risk classifier for municipalities in Kosovo \
to non-expert readers. You receive one prediction as JSON.

Rules:
1. The risk level was decided by the ML model. Never change it, never suggest a different level, \
and never make your own risk assessment.
2. The first line of your answer must be exactly: "Risk level: {risk_level}" (in English, as written).
3. Then write 2-3 short paragraphs in {language}:
   - what the predicted level means in plain words;
   - which input values stand out, using the feature-importance ranking only as "features the model \
relies on most overall" - not as reasons for this specific prediction and never as causes of flooding;
   - the main limitations and a short caution.
4. class_scores are shares of the model's trees voting for each class. Do not call them probabilities \
of flooding or percentages of risk.
5. If warnings are present, mention that the prediction is less reliable because some values are outside \
the data the model was trained on.
6. Use only the information in the JSON. Do not invent data, historical floods, or statistics.
7. No headings, no bullet points, no markdown. Maximum 200 words after the first line.
"""


def build_context(assessment: FloodRiskAssessment, metadata: ClassifierMetadata) -> dict:
    """The complete data sent to OpenAI for one explanation."""
    data = assessment.input
    return {
        "municipality": data.municipality,
        "predicted_risk_level": assessment.classification.risk_level.label,
        "class_scores": {level.label: score for level, score in assessment.classification.class_scores.items()},
        "input_values": {
            name: {"value": value, "unit": metadata.feature_units.get(name)}
            for name, value in data.features().items()
        },
        "warnings": assessment.warnings,
        "model": {
            "type": metadata.model_name,
            "cross_validated_macro_f1": f"{metadata.cv_macro_f1_mean} ± {metadata.cv_macro_f1_std}",
            "features_the_model_relies_on_most": list(metadata.feature_importances)[:4],
            "limitations": metadata.limitations,
        },
    }


class OpenAIRiskInterpreter(RiskInterpreter):
    def __init__(
        self,
        api_key: str,
        model: str,
        language: str = "English",
        timeout_seconds: float = 30.0,
        max_output_tokens: int = 600,
    ) -> None:
        self._client = openai.OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=1)
        self._model = model
        self._language = language
        self._max_output_tokens = max_output_tokens

    def interpret(self, assessment: FloodRiskAssessment, metadata: ClassifierMetadata) -> str:
        instructions = INSTRUCTIONS.format(
            risk_level=assessment.classification.risk_level.label, language=self._language
        )
        context = json.dumps(build_context(assessment, metadata), ensure_ascii=False, indent=2)
        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=instructions,
                input=context,
                max_output_tokens=self._max_output_tokens,
                store=False,  # do not keep the request/response on OpenAI's side for later retrieval
            )
        except openai.OpenAIError as exc:
            raise InterpretationUnavailable(f"{type(exc).__name__}: {exc}") from exc

        text = response.output_text
        if not text or not text.strip():
            raise InterpretationUnavailable("OpenAI returned an empty response.")
        return text
