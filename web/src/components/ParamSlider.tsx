interface ParamSliderProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  unit?: string;
}

export default function ParamSlider({
  label,
  value,
  onChange,
  min,
  max,
  step,
  disabled,
  unit,
}: ParamSliderProps) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-text-secondary">
          {label}
        </label>
        <span className="text-xs tabular-nums text-text-primary">
          {value}
          {unit ? ` ${unit}` : ""}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        disabled={disabled}
        className="w-full disabled:opacity-50"
      />
    </div>
  );
}
