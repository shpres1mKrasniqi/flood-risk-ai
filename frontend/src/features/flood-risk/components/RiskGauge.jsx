import { useTranslation } from "../../../i18n/useTranslation";
const LEVELS = ["Low", "Medium", "High"];
const COLORS = {
  Low: "var(--color-risk-low)",
  Medium: "var(--color-risk-medium)",
  High: "var(--color-risk-high)",
};
const WATER_HEIGHT = { Low: 1 / 3, Medium: 2 / 3, High: 0.97 };

const TOP = 16;
const BOTTOM = 284;
const POLE_X = 36;
const POLE_W = 44;
const HEIGHT = BOTTOM - TOP;
const BAND = HEIGHT / 3;

export default function RiskGauge({ risk, scores }) {
  const { t, formatNumber } = useTranslation();
  const waterTop = BOTTOM - HEIGHT * WATER_HEIGHT[risk];
  const ticks = Array.from({ length: 13 }, (_, i) => TOP + (HEIGHT / 12) * i);

  return (
    <svg
      viewBox="0 0 260 300"
      className="block h-auto w-full"
      role="img"
      aria-label={t("result.gaugeLabel", { level: t(`risk.${risk}`) })}
    >
      <g key={risk}>
        <rect
          className="water-rise"
          x="0"
          y={waterTop}
          width={POLE_X + POLE_W + 18}
          height={BOTTOM - waterTop}
          fill="var(--color-river)"
          opacity="0.28"
        />
        <line
          className="water-rise"
          x1="0"
          x2={POLE_X + POLE_W + 18}
          y1={waterTop}
          y2={waterTop}
          stroke="var(--color-river)"
          strokeWidth="2"
        />
      </g>
      <rect
        x={POLE_X}
        y={TOP}
        width={POLE_W}
        height={HEIGHT}
        fill="var(--color-surface)"
        stroke="var(--color-ink)"
      />
      {ticks.map((y, i) => (
        <line
          key={y}
          x1={POLE_X}
          x2={POLE_X + (i % 4 === 0 ? POLE_W : POLE_W / 2)}
          y1={y}
          y2={y}
          stroke="var(--color-ink)"
          strokeWidth={i % 4 === 0 ? 2 : 1}
        />
      ))}

      {LEVELS.map((level, i) => {
        const y = BOTTOM - BAND * (i + 1);
        const active = level === risk;
        return (
          <g key={level}>
            <rect
              x={POLE_X + POLE_W}
              y={y}
              width="10"
              height={BAND}
              fill={COLORS[level]}
            />
            <text
              x={POLE_X + POLE_W + 26}
              y={y + BAND / 2 - 4}
              fontSize="17"
              fontWeight={active ? 800 : 500}
              fill={active ? "var(--color-ink)" : "var(--color-muted)"}
            >
              {t(`risk.${level}`)}
            </text>
            <text
              x={POLE_X + POLE_W + 26}
              y={y + BAND / 2 + 16}
              fontSize="14"
              fill="var(--color-muted)"
            >
              {formatNumber((scores[level] ?? 0) * 100, 1)}%
            </text>
          </g>
        );
      })}
    </svg>
  );
}
