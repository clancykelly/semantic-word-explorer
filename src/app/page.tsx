"use client";

import { useRouter } from "next/navigation";
import { useCallback } from "react";
import { SearchInput } from "@/components/SearchInput";
import { getAvailableWords } from "@/lib/mock-data";

export default function Home() {
  const router = useRouter();

  const handleSearch = useCallback(
    (word: string) => {
      router.push(`/explore/${encodeURIComponent(word)}`);
    },
    [router]
  );

  const availableWords = getAvailableWords();

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4 py-16 bg-gradient-to-b from-zinc-50 to-white dark:from-zinc-950 dark:to-zinc-900">
      <main className="w-full max-w-3xl flex flex-col items-center gap-8">
        {/* Hero */}
        <div className="text-center space-y-4">
          <h1 className="text-4xl md:text-5xl font-bold text-zinc-900 dark:text-zinc-100 tracking-tight">
            Semantic Word Explorer
          </h1>
          <p className="text-lg md:text-xl text-zinc-600 dark:text-zinc-400 max-w-xl mx-auto">
            Discover unexpected words through semantic relationships.
            Go beyond synonyms to find conceptually connected vocabulary.
          </p>
        </div>

        {/* Search */}
        <SearchInput onSearch={handleSearch} />

        {/* Example words */}
        <div className="text-center">
          <p className="text-sm text-zinc-500 dark:text-zinc-500 mb-3">
            Try exploring:
          </p>
          <div className="flex flex-wrap gap-2 justify-center">
            {availableWords.map((word) => (
              <button
                key={word}
                onClick={() => handleSearch(word)}
                className="px-4 py-1.5 rounded-full text-sm font-medium
                  bg-zinc-100 dark:bg-zinc-800
                  text-zinc-700 dark:text-zinc-300
                  hover:bg-indigo-100 dark:hover:bg-indigo-900/30
                  hover:text-indigo-700 dark:hover:text-indigo-300
                  transition-colors duration-200"
              >
                {word}
              </button>
            ))}
          </div>
        </div>

        {/* Features */}
        <div className="grid md:grid-cols-3 gap-6 mt-12 w-full">
          <div className="text-center p-6 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800">
            <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center">
              <svg
                className="w-6 h-6 text-indigo-600 dark:text-indigo-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 10V3L4 14h7v7l9-11h-7z"
                />
              </svg>
            </div>
            <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 mb-2">
              Beyond Synonyms
            </h3>
            <p className="text-sm text-zinc-600 dark:text-zinc-400">
              Find conceptually related words, not just direct synonyms
            </p>
          </div>

          <div className="text-center p-6 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800">
            <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-pink-100 dark:bg-pink-900/30 flex items-center justify-center">
              <svg
                className="w-6 h-6 text-pink-600 dark:text-pink-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
            </div>
            <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 mb-2">
              Visual Clusters
            </h3>
            <p className="text-sm text-zinc-600 dark:text-zinc-400">
              See word relationships grouped by meaning in an interactive plot
            </p>
          </div>

          <div className="text-center p-6 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800">
            <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-teal-100 dark:bg-teal-900/30 flex items-center justify-center">
              <svg
                className="w-6 h-6 text-teal-600 dark:text-teal-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01"
                />
              </svg>
            </div>
            <h3 className="font-semibold text-zinc-900 dark:text-zinc-100 mb-2">
              Sense Disambiguation
            </h3>
            <p className="text-sm text-zinc-600 dark:text-zinc-400">
              Explore different meanings of polysemous words separately
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
