"use client";

import { useState, useCallback, useEffect, useRef, FormEvent } from "react";

interface SearchInputProps {
  onSearch: (word: string) => void;
  isLoading?: boolean;
  initialValue?: string;
}

export function SearchInput({
  onSearch,
  isLoading = false,
  initialValue = "",
}: SearchInputProps) {
  const [value, setValue] = useState(initialValue);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Keep the field in sync when navigating between words
  useEffect(() => {
    setValue(initialValue);
  }, [initialValue]);

  // "/" focuses the search field from anywhere
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "/") return;
      const t = e.target as HTMLElement | null;
      if (
        t &&
        (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)
      )
        return;
      e.preventDefault();
      inputRef.current?.focus();
      inputRef.current?.select();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  const validateAndSubmit = useCallback(
    (input: string) => {
      const trimmed = input.trim();
      if (!trimmed) {
        setError("type a word first");
        return;
      }
      if (trimmed.includes(" ")) {
        setError("single words only");
        return;
      }
      if (!/^[a-zA-Z-]+$/.test(trimmed)) {
        setError("letters only");
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

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="flex items-center gap-2 border-b border-[var(--line)] focus-within:border-[var(--ink)] transition-colors">
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            if (error) setError(null);
          }}
          placeholder="type a word…"
          disabled={isLoading}
          className="min-w-0 flex-1 bg-transparent py-2 text-[15px] outline-none placeholder:text-[var(--faint)] disabled:opacity-50"
          aria-label="Search word"
          aria-invalid={!!error}
        />
        <kbd className="hidden md:inline-block font-mono text-[10px] leading-none text-[var(--faint)] border border-[var(--line)] rounded-[3px] px-1.5 py-1">
          /
        </kbd>
        <button
          type="submit"
          disabled={isLoading || !value.trim()}
          className="font-mono text-[11px] uppercase tracking-[0.14em] px-2 py-1.5 text-[var(--muted)] hover:text-[var(--accent)] transition-colors disabled:opacity-40 disabled:hover:text-[var(--muted)]"
        >
          {isLoading ? "…" : "go"}
        </button>
      </div>
      {error && (
        <p className="mt-1.5 font-mono text-[11px] text-[var(--accent)]" role="alert">
          {error}
        </p>
      )}
    </form>
  );
}
