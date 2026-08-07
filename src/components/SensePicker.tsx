"use client";

import type { WordSense } from "@/lib/types";

interface SensePickerProps {
  senses: WordSense[];
  selectedSense: string; // "" = all senses
  onSelectSense: (sense: string) => void;
}

export function SensePicker({
  senses,
  selectedSense,
  onSelectSense,
}: SensePickerProps) {
  if (senses.length <= 1) {
    return null;
  }

  const options: { sense: string; label: string }[] = [
    { sense: "", label: "all" },
    ...senses.map((s) => ({ sense: s.sense, label: s.label })),
  ];

  return (
    <div className="flex flex-wrap items-center justify-center gap-1.5">
      <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-[var(--faint)] mr-1.5">
        sense
      </span>
      {options.map((option) => {
        const active = selectedSense === option.sense;
        return (
          <button
            key={option.sense || "all"}
            onClick={() => onSelectSense(option.sense)}
            title={option.label}
            aria-pressed={active}
            className={`font-mono text-[11px] lowercase px-2.5 py-1 rounded-[3px] border transition-colors max-w-[320px] truncate ${
              active
                ? "bg-[var(--ink)] text-[var(--paper)] border-transparent"
                : "border-[var(--line)] text-[var(--muted)] hover:text-[var(--ink)] hover:border-[var(--muted)]"
            }`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
