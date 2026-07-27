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
  // Only worth showing when the word splits into 2+ meaning groups.
  if (senses.length <= 1) {
    return null;
  }

  const options: { sense: string; label: string }[] = [
    { sense: "", label: "All" },
    ...senses.map((s) => ({ sense: s.sense, label: s.label })),
  ];

  return (
    <div className="flex flex-wrap gap-2 justify-center">
      <span className="text-sm text-zinc-500 dark:text-zinc-400 self-center mr-1">
        Meaning:
      </span>
      {options.map((option) => (
        <button
          key={option.sense || "all"}
          onClick={() => onSelectSense(option.sense)}
          className={`
            px-4 py-1.5 rounded-full text-sm font-medium
            transition-all duration-200
            ${
              selectedSense === option.sense
                ? "bg-indigo-600 text-white shadow-md"
                : "bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-700"
            }
          `}
          aria-pressed={selectedSense === option.sense}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
