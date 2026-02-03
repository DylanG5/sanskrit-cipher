/**
 * Validation utilities for fragment metadata editing
 */

import { SCRIPT_TYPES } from '../types/constants';

export interface ValidationResult {
  valid: boolean;
  error?: string;
}

export function validateLineCount(value: number): ValidationResult {
  // Allow null/undefined to clear the field
  if (value === null || value === undefined || value === '') {
    return { valid: true };
  }
  const num = Number(value);
  if (isNaN(num)) {
    return { valid: false, error: 'Line count must be a number' };
  }
  if (!Number.isInteger(num)) {
    return { valid: false, error: 'Line count must be a whole number' };
  }
  if (num < 0) {
    return { valid: false, error: 'Line count cannot be negative' };
  }
  if (num > 100) {
    return { valid: false, error: 'Line count must be 100 or less' };
  }
  return { valid: true };
}

export function validatePixelsPerUnit(value: number): ValidationResult {
  if (isNaN(value)) {
    return { valid: false, error: 'Must be a valid number' };
  }
  if (value <= 0) {
    return { valid: false, error: 'Pixels per unit must be positive' };
  }
  if (value > 10000) {
    return { valid: false, error: 'Value seems too large' };
  }
  return { valid: true };
}

export function validateScriptType(value: string): ValidationResult {
  // Allow empty/undefined to clear the field
  if (value === undefined || value === null) {
    return { valid: true };
  }
  const trimmed = String(value).trim();
  if (trimmed.length === 0) {
    return { valid: true }; // Allow clearing the field
  }
  // Check if the script type is in the allowed list
  if (!SCRIPT_TYPES.includes(trimmed as any)) {
    return { valid: false, error: 'Please select a valid script type from the dropdown' };
  }
  return { valid: true };
}

export function validateScaleUnit(value: string): ValidationResult {
  if (!['cm', 'mm'].includes(value)) {
    return { valid: false, error: 'Scale unit must be "cm" or "mm"' };
  }
  return { valid: true };
}
