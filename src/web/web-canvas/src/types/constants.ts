/**
 * Constants for fragment metadata
 */

/**
 * Available Turkestan Brāhmī script types
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

export type ScriptType = typeof SCRIPT_TYPES[number];
