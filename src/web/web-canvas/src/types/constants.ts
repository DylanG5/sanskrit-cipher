/**
 * Constants for fragment metadata
 */

/**
 * Model output values (stored in database without accents)
 */
export const MODEL_SCRIPT_TYPES = [
  'Early South Turkestan Brahmi',
  'North Turkestan Brahmi',
  'North Turkestan Brahmi, type a',
  'South Turkestan Brahmi',
  'Turkestan Gupta Type',
] as const;

/**
 * Display values with proper diacritics for UI
 */
export const SCRIPT_TYPES = [
  'Early South Turkestan Brāhmī',
  'South Turkestan Brāhmī (main type)',
  'Turkestan Gupta Type',
  'North Turkestan Brāhmī',
  'North Turkestan Brāhmī, type a',
  'North Turkestan Brāhmī, type b',
  'Early Turkestan Brāhmī, alphabet r',
] as const;

/**
 * Mapping from model output (database values) to display values (with diacritics)
 */
export const SCRIPT_TYPE_DISPLAY_MAP: Record<string, string> = {
  'Early South Turkestan Brahmi': 'Early South Turkestan Brāhmī',
  'North Turkestan Brahmi': 'North Turkestan Brāhmī',
  'North Turkestan Brahmi, type a': 'North Turkestan Brāhmī, type a',
  'South Turkestan Brahmi': 'South Turkestan Brāhmī (main type)',
  'Turkestan Gupta Type': 'Turkestan Gupta Type',
  // Legacy/manual entries (keep for backwards compatibility)
  'North Turkestan Brāhmī, type b': 'North Turkestan Brāhmī, type b',
  'Early Turkestan Brāhmī, alphabet r': 'Early Turkestan Brāhmī, alphabet r',
};

/**
 * Reverse mapping from display values to database values
 */
export const SCRIPT_TYPE_DB_MAP: Record<string, string> = {
  'Early South Turkestan Brāhmī': 'Early South Turkestan Brahmi',
  'North Turkestan Brāhmī': 'North Turkestan Brahmi',
  'North Turkestan Brāhmī, type a': 'North Turkestan Brahmi, type a',
  'South Turkestan Brāhmī (main type)': 'South Turkestan Brahmi',
  'Turkestan Gupta Type': 'Turkestan Gupta Type',
  // Legacy/manual entries
  'North Turkestan Brāhmī, type b': 'North Turkestan Brāhmī, type b',
  'Early Turkestan Brāhmī, alphabet r': 'Early Turkestan Brāhmī, alphabet r',
};

/**
 * Convert database script type to display format
 */
export function getScriptTypeDisplay(dbValue: string | null | undefined): string | undefined {
  if (!dbValue) return undefined;
  return SCRIPT_TYPE_DISPLAY_MAP[dbValue] || dbValue;
}

/**
 * Convert display script type to database format
 */
export function getScriptTypeDB(displayValue: string | null | undefined): string | undefined {
  if (!displayValue) return undefined;
  return SCRIPT_TYPE_DB_MAP[displayValue] || displayValue;
}

export type ScriptType = typeof SCRIPT_TYPES[number];
