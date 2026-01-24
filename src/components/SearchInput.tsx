"use client";

import { useState, useCallback, FormEvent } from "react";

interface SearchInputProps {
  onSearch: (word: string) => void;
  isLoading?: boolean;
  initialValue?: string;
  didYouMean?: string;
  onDidYouMeanClick?: (word: string) => void;
}

export function SearchInput({
  onSearch,
  isLoading = false,
  initialValue = "",
  didYouMean,
  onDidYouMeanClick,
}: SearchInputProps) {
  const [value, setValue] = useState(initialValue);
  const [error, setError] = useState<string | null>(null);

  const validateAndSubmit = useCallback(
    (input: string) => {
      const trimmed = input.trim();

      // Check for empty input
      if (!trimmed) {
        setError("Please enter a word");
        return;
      }

      // Check for multiple words
      if (trimmed.includes(" ")) {
        setError("Please enter a single word");
        return;
      }

      // Check for non-alphabetic characters (allow hyphens for compound words)
      if (!/^[a-zA-Z-]+$/.test(trimmed)) {
        setError("Please enter only letters");
        return;
      }

      setError(null);
      onSearch(trimmed.toLowerCase());
    },
    [onSearch]
  );

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    validateAndSubmit(value);
  };

  const handleDidYouMeanClick = () => {
    if (didYouMean && onDidYouMeanClick) {
      setValue(didYouMean);
      onDidYouMeanClick(didYouMean);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-xl">
      <div className="relative">
        <input
          type="text"
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            if (error) setError(null);
          }}
          placeholder="Enter a word to explore..."
          disabled={isLoading}
          className={`
            w-full px-6 py-4 text-lg rounded-full
            border-2 transition-all duration-200
            bg-white dark:bg-zinc-900
            text-zinc-900 dark:text-zinc-100
            placeholder:text-zinc-400 dark:placeholder:text-zinc-500
            focus:outline-none focus:ring-2 focus:ring-offset-2
            ${
              error
                ? "border-red-400 focus:border-red-500 focus:ring-red-500"
                : "border-zinc-200 dark:border-zinc-700 focus:border-indigo-500 focus:ring-indigo-500"
            }
            ${isLoading ? "opacity-70 cursor-not-allowed" : ""}
          `}
          aria-label="Search word"
          aria-invalid={!!error}
          aria-describedby={error ? "search-error" : undefined}
        />
        <button
          type="submit"
          disabled={isLoading || !value.trim()}
          className={`
            absolute right-2 top-1/2 -translate-y-1/2
            px-6 py-2.5 rounded-full font-medium
            transition-all duration-200
            ${
              isLoading || !value.trim()
                ? "bg-zinc-200 dark:bg-zinc-700 text-zinc-400 cursor-not-allowed"
                : "bg-indigo-600 hover:bg-indigo-700 text-white"
            }
          `}
          aria-label={isLoading ? "Searching..." : "Search"}
        >
          {isLoading ? (
            <span className="flex items-center gap-2">
              <svg
                className="animate-spin h-4 w-4"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              Exploring
            </span>
          ) : (
            "Explore"
          )}
        </button>
      </div>

      {/* Error message */}
      {error && (
        <p
          id="search-error"
          className="mt-2 text-sm text-red-500 dark:text-red-400 pl-4"
          role="alert"
        >
          {error}
        </p>
      )}

      {/* Did you mean suggestion */}
      {didYouMean && !error && (
        <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400 pl-4">
          Did you mean{" "}
          <button
            type="button"
            onClick={handleDidYouMeanClick}
            className="text-indigo-600 dark:text-indigo-400 hover:underline font-medium"
          >
            {didYouMean}
          </button>
          ?
        </p>
      )}
    </form>
  );
}
