"use client";

import type { WordSense } from "@/lib/types";

interface SensePickerProps {
  senses: WordSense[];
  selectedSense: string;
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

  return (
    <div className="flex flex-wrap gap-2 justify-center">
      <span className="text-sm text-zinc-500 dark:text-zinc-400 self-center mr-1">
        Meaning:
      </span>
      {senses.map((sense) => (
        <button
          key={sense.sense}
          onClick={() => onSelectSense(sense.sense)}
          className={`
            px-4 py-1.5 rounded-full text-sm font-medium
            transition-all duration-200
            ${
              selectedSense === sense.sense
                ? "bg-indigo-600 text-white shadow-md"
                : "bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-700"
            }
          `}
          aria-pressed={selectedSense === sense.sense}
        >
          {sense.label}
        </button>
      ))}
    </div>
  );
}
