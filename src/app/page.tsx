"use client";

import { useRouter } from "next/navigation";
import { useCallback } from "react";
import { SearchInput } from "@/components/SearchInput";

// Words that show the instrument off — including a polysemous one and a preposition.
const EXAMPLE_WORDS = ["happy", "bank", "run", "ocean", "bright", "beneath"];

export default function Home() {
  const router = useRouter();

  const handleSearch = useCallback(
    (word: string) => {
      router.push(`/explore/${encodeURIComponent(word)}`);
    },
    [router]
  );

  return (
    <div className="min-h-screen flex flex-col">
      <main className="flex-1 flex flex-col items-center justify-center px-6 py-20">
        <div className="w-full max-w-xl space-y-10">
          <div className="space-y-4">
            <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-[var(--faint)]">
              semantic word explorer
            </p>
            <h1 className="text-[44px] md:text-[56px] leading-[1.02] font-bold tracking-[-0.03em]">
              The word you
              <br />
              almost have.
            </h1>
            <p className="font-mono text-[12px] leading-relaxed text-[var(--muted)] max-w-md">
              senses, synonyms, and the neighborhood of meaning around any word —
              weighted by closeness, marked for register and rarity.
            </p>
          </div>

          <SearchInput onSearch={handleSearch} />

          <div className="flex flex-wrap items-baseline gap-x-4 gap-y-2">
            <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-[var(--faint)]">
              try
            </span>
            {EXAMPLE_WORDS.map((word) => (
              <button
                key={word}
                onClick={() => handleSearch(word)}
                className="text-[15px] text-[var(--muted)] hover:text-[var(--accent)] transition-colors"
              >
                {word}
              </button>
            ))}
          </div>
        </div>
      </main>

      <footer className="px-6 py-5">
        <p className="max-w-xl mx-auto font-mono text-[10px] uppercase tracking-[0.14em] text-[var(--faint)]">
          / to search · click copies · ↗ explores
        </p>
      </footer>
    </div>
  );
}
