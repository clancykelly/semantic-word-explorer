/**
 * Prevention Strategies - Utility Functions and Type Guards
 *
 * This module provides reusable utilities for implementing the prevention
 * strategies documented in PREVENTION_STRATEGIES.md.
 *
 * Key Issues Addressed:
 * 1. API Score Normalization
 * 2. Additive Filter Design
 * 3. Distant Relationship Pollution
 * 4. Visualization Collision
 */

import type { WordNeighbor, Coordinates } from "./types";

// =============================================================================
// 1. API Score Normalization
// =============================================================================

/**
 * Normalize scores within a single source to 0-1 range.
 *
 * @param items - Array of items with score property
 * @param scoreKey - Key to access the score value
 * @returns Items with normalized scores
 */
export function normalizeScores<T extends { score?: number }>(
  items: T[]
): (T & { normalizedScore: number })[] {
  if (items.length === 0) return [];

  const scores = items.map((item) => item.score ?? 0);
  const maxScore = Math.max(...scores);
  const minScore = Math.min(...scores);
  const range = maxScore - minScore;

  if (range === 0) {
    // All same score - assign based on position
    return items.map((item, index) => ({
      ...item,
      normalizedScore: 1 - index / items.length,
    }));
  }

  return items.map((item) => ({
    ...item,
    normalizedScore: ((item.score ?? 0) - minScore) / range,
  }));
}

/**
 * Validate that all similarity scores are in valid range.
 *
 * @param neighbors - Array of word neighbors to validate
 * @throws Error if any score is outside [0, 1] range
 */
export function validateScoreRange(neighbors: WordNeighbor[]): void {
  for (const neighbor of neighbors) {
    if (neighbor.similarity < 0 || neighbor.similarity > 1) {
      throw new Error(
        `Score out of range for "${neighbor.word}": ${neighbor.similarity}`
      );
    }
  }
}

// =============================================================================
// 2. Additive Filter Design
// =============================================================================

/**
 * Type for additive filter configuration.
 * All "include X" filters must be additive - the enabled state
 * always includes everything from the disabled state.
 */
export interface AdditiveFilterConfig {
  /** Include rare/uncommon words */
  includeRare: boolean;
}

/**
 * Verify that one set is a superset of another (for testing additive filters).
 *
 * @param superset - The larger set (e.g., results with includeRare=true)
 * @param subset - The smaller set (e.g., results with includeRare=false)
 * @returns Object indicating whether superset property holds and any missing items
 */
export function verifySupersetRelationship<T>(
  superset: Set<T>,
  subset: Set<T>
): { isValid: boolean; missing: T[] } {
  const missing: T[] = [];

  for (const item of subset) {
    if (!superset.has(item)) {
      missing.push(item);
    }
  }

  return {
    isValid: missing.length === 0,
    missing,
  };
}

/**
 * Filter implementation that guarantees additive behavior.
 *
 * @param items - Full list of items
 * @param baseFilter - Predicate for base set (e.g., common words)
 * @param includeExtra - Whether to include items that don't match baseFilter
 * @param baseLimit - Maximum items from base set
 * @param extraLimit - Maximum extra items when includeExtra is true
 * @returns Filtered items with superset guarantee
 */
export function additiveFilter<T>(
  items: T[],
  baseFilter: (item: T) => boolean,
  includeExtra: boolean,
  baseLimit: number = 100,
  extraLimit: number = 50
): T[] {
  const baseItems = items.filter(baseFilter).slice(0, baseLimit);

  if (!includeExtra) {
    return baseItems;
  }

  // Additive: base items + extra items
  const extraItems = items.filter((item) => !baseFilter(item)).slice(0, extraLimit);
  return [...baseItems, ...extraItems];
}

// =============================================================================
// 3. Distant Relationship Pollution
// =============================================================================

/**
 * Configuration for relationship filtering.
 */
export interface RelationshipConfig {
  /** Minimum similarity score to include (0-1) */
  minSimilarity: number;
  /** Maximum results per relationship type */
  maxResults: number;
}

/**
 * Default limits for different relationship types.
 * More direct relationships get higher limits.
 */
export const RELATIONSHIP_LIMITS: Record<string, number> = {
  synonym: 100,
  meansLike: 150,
  similarTo: 80,
  triggers: 30,
  antonym: 50,
};

/**
 * Filter out distantly related results below a similarity threshold.
 *
 * @param neighbors - Array of word neighbors
 * @param minSimilarity - Minimum similarity to include (default 0.05)
 * @returns Filtered neighbors with meaningful relationships
 */
export function filterDistantRelationships(
  neighbors: WordNeighbor[],
  minSimilarity: number = 0.05
): WordNeighbor[] {
  return neighbors.filter(
    (neighbor) => neighbor.similarity >= minSimilarity
  );
}

/**
 * Check if a word should be excluded (multi-word phrases, etc.).
 *
 * @param word - Word to check
 * @returns true if word should be excluded
 */
export function shouldExcludeWord(word: string): boolean {
  // Exclude multi-word phrases
  if (word.includes(" ")) return true;

  // Exclude very short words (likely noise)
  if (word.length < 2) return true;

  return false;
}

// =============================================================================
// 4. Visualization Collision
// =============================================================================

/**
 * Configuration for collision avoidance.
 */
export interface CollisionConfig {
  /** Minimum distance between points */
  minDistance: number;
  /** Number of iterations for collision resolution */
  iterations: number;
  /** Protected radius around center (query word) */
  centerProtectedRadius: number;
}

/**
 * Default collision avoidance configuration.
 */
export const DEFAULT_COLLISION_CONFIG: CollisionConfig = {
  minDistance: 0.04,
  iterations: 20,
  centerProtectedRadius: 0.04,
};

/**
 * Calculate Euclidean distance between two points.
 */
export function distance(p1: Coordinates, p2: Coordinates): number {
  return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
}

/**
 * Check if coordinates have any collisions.
 *
 * @param coordinates - Array of coordinate objects
 * @param minDistance - Minimum allowed distance
 * @returns Array of collision pairs with their distances
 */
export function findCollisions(
  coordinates: Coordinates[],
  minDistance: number = 0.03
): Array<{ i: number; j: number; distance: number }> {
  const collisions: Array<{ i: number; j: number; distance: number }> = [];

  for (let i = 0; i < coordinates.length; i++) {
    for (let j = i + 1; j < coordinates.length; j++) {
      const d = distance(coordinates[i], coordinates[j]);
      if (d < minDistance) {
        collisions.push({ i, j, distance: d });
      }
    }
  }

  return collisions;
}

/**
 * Check if a point is too close to the center.
 *
 * @param coord - Coordinate to check
 * @param minDistance - Minimum distance from center
 * @returns true if point is too close to center
 */
export function isTooCloseToCenter(
  coord: Coordinates,
  minDistance: number = 0.04
): boolean {
  const center: Coordinates = { x: 0.5, y: 0.5 };
  return distance(coord, center) < minDistance;
}

/**
 * Apply collision avoidance to a set of coordinates.
 * This is a client-side implementation that can be used as a backup
 * if server-side collision avoidance is insufficient.
 *
 * @param coordinates - Initial coordinates
 * @param config - Collision avoidance configuration
 * @returns Adjusted coordinates with reduced collisions
 */
export function avoidCollisions(
  coordinates: Coordinates[],
  config: CollisionConfig = DEFAULT_COLLISION_CONFIG
): Coordinates[] {
  if (coordinates.length < 2) return coordinates;

  const coords = coordinates.map((c) => ({ x: c.x, y: c.y }));
  const n = coords.length;
  const center: Coordinates = { x: 0.5, y: 0.5 };

  for (let iter = 0; iter < config.iterations; iter++) {
    // Push away from center
    for (let i = 0; i < n; i++) {
      const dx = coords[i].x - center.x;
      const dy = coords[i].y - center.y;
      const d = Math.sqrt(dx * dx + dy * dy);

      if (d < config.centerProtectedRadius && d > 0.001) {
        const push = (config.centerProtectedRadius - d) * 0.5;
        coords[i].x += (dx / d) * push;
        coords[i].y += (dy / d) * push;
      }
    }

    // Push points apart from each other
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = coords[j].x - coords[i].x;
        const dy = coords[j].y - coords[i].y;
        const d = Math.sqrt(dx * dx + dy * dy);

        if (d < config.minDistance && d > 0.001) {
          const overlap = (config.minDistance - d) * 0.3;
          const dxNorm = dx / d;
          const dyNorm = dy / d;

          coords[i].x -= dxNorm * overlap;
          coords[i].y -= dyNorm * overlap;
          coords[j].x += dxNorm * overlap;
          coords[j].y += dyNorm * overlap;
        }
      }
    }
  }

  // Clamp to valid range
  return coords.map((c) => ({
    x: Math.max(0.02, Math.min(0.98, c.x)),
    y: Math.max(0.02, Math.min(0.98, c.y)),
  }));
}

// =============================================================================
// Testing Utilities
// =============================================================================

/**
 * Test helper to verify prevention strategies are properly implemented.
 */
export const PreventionStrategyTests = {
  /**
   * Test that all similarity scores are in valid range.
   */
  testScoreNormalization(neighbors: WordNeighbor[]): {
    passed: boolean;
    failures: string[];
  } {
    const failures: string[] = [];

    for (const neighbor of neighbors) {
      if (neighbor.similarity < 0 || neighbor.similarity > 1) {
        failures.push(
          `"${neighbor.word}": similarity ${neighbor.similarity} out of [0,1] range`
        );
      }
    }

    return { passed: failures.length === 0, failures };
  },

  /**
   * Test that includeRare=true is a superset of includeRare=false.
   */
  testAdditiveFilter(
    commonResults: WordNeighbor[],
    rareResults: WordNeighbor[]
  ): { passed: boolean; missingWords: string[] } {
    const commonWords = new Set(commonResults.map((n) => n.word));
    const rareWords = new Set(rareResults.map((n) => n.word));

    const result = verifySupersetRelationship(rareWords, commonWords);

    return {
      passed: result.isValid,
      missingWords: result.missing as string[],
    };
  },

  /**
   * Test that no coordinates are too close together.
   */
  testCollisionAvoidance(
    neighbors: WordNeighbor[],
    minDistance: number = 0.03
  ): { passed: boolean; collisions: Array<{ word1: string; word2: string; distance: number }> } {
    const collisionDetails: Array<{ word1: string; word2: string; distance: number }> = [];
    const coordinates = neighbors.map((n) => n.coordinates);

    for (let i = 0; i < coordinates.length; i++) {
      for (let j = i + 1; j < coordinates.length; j++) {
        const d = distance(coordinates[i], coordinates[j]);
        if (d < minDistance) {
          collisionDetails.push({
            word1: neighbors[i].word,
            word2: neighbors[j].word,
            distance: d,
          });
        }
      }
    }

    return {
      passed: collisionDetails.length === 0,
      collisions: collisionDetails,
    };
  },

  /**
   * Test that results don't include distantly related words.
   */
  testDistantRelationships(
    neighbors: WordNeighbor[],
    queryWord: string,
    minSimilarity: number = 0.05
  ): { passed: boolean; distantWords: Array<{ word: string; similarity: number }> } {
    const distantWords: Array<{ word: string; similarity: number }> = [];

    for (const neighbor of neighbors) {
      if (neighbor.word === queryWord) continue;
      if (neighbor.similarity < minSimilarity) {
        distantWords.push({
          word: neighbor.word,
          similarity: neighbor.similarity,
        });
      }
    }

    return {
      passed: distantWords.length === 0,
      distantWords,
    };
  },
};
