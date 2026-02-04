import React, { useState, useEffect, useRef } from 'react';
import { ManuscriptFragment, CanvasFragment } from '../types/fragment';
import { updateFragmentMetadata, getFragmentById } from '../services/fragment-service';
import {
  validateLineCount,
  validatePixelsPerUnit,
  validateScriptType,
  validateScaleUnit,
} from '../utils/metadataValidation';
import { SCRIPT_TYPES, getScriptTypeDB } from '../types/constants';

interface FragmentMetadataProps {
  fragment: ManuscriptFragment;
  onClose: () => void;
  onUpdate?: () => void; // Callback to refresh fragment data after edit
  canvasFragment?: CanvasFragment | null; // Canvas fragment data for resize-based scale calculation
  gridScale?: number; // Grid scale in pixels per cm
}

const FragmentMetadata: React.FC<FragmentMetadataProps> = ({
  fragment: initialFragment,
  onClose,
  onUpdate,
  canvasFragment,
  gridScale = 25
}) => {
  const [fragment, setFragment] = useState(initialFragment);
  const { metadata } = fragment;

  // Update local fragment when prop changes
  useEffect(() => {
    setFragment(initialFragment);
  }, [initialFragment]);

  // Edit state management
  const [editingField, setEditingField] = useState<string | null>(null);
  const [editValues, setEditValues] = useState<Record<string, any>>({});
  const [savingField, setSavingField] = useState<string | null>(null);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [saveSuccess, setSaveSuccess] = useState<string | null>(null);

  const startEditing = (field: string, currentValue: any) => {
    setEditingField(field);
    setEditValues({ ...editValues, [field]: currentValue });
    setFieldErrors({ ...fieldErrors, [field]: '' });
  };

  const cancelEditing = () => {
    setEditingField(null);
    setFieldErrors({});
  };

  const saveField = async (field: string, dbField: string, value: any) => {
    // Validate based on field type
    let validation;
    if (field === 'lineCount') {
      validation = validateLineCount(Number(value));
    } else if (field === 'pixelsPerUnit') {
      validation = validatePixelsPerUnit(Number(value));
    } else if (field === 'script') {
      validation = validateScriptType(String(value));
    } else if (field === 'scaleUnit') {
      validation = validateScaleUnit(String(value));
    } else {
      validation = { valid: true };
    }

    if (!validation.valid) {
      setFieldErrors({ ...fieldErrors, [field]: validation.error || 'Invalid value' });
      return;
    }

    setSavingField(field);
    setFieldErrors({ ...fieldErrors, [field]: '' });

    // Convert display value to database value for script type
    let dbValue = value;
    if (field === 'script') {
      dbValue = getScriptTypeDB(value) || value;
    }

    try {
      const result = await updateFragmentMetadata(fragment.id, { [dbField]: dbValue });

      if (result.success) {
        // Refetch the fragment to get updated data
        const updatedFragment = await getFragmentById(fragment.id);
        if (updatedFragment) {
          setFragment(updatedFragment);
        }

        setSaveSuccess(field);
        setTimeout(() => setSaveSuccess(null), 2000);
        setEditingField(null);

        if (onUpdate) {
          onUpdate();
        }
      } else {
        setFieldErrors({ ...fieldErrors, [field]: result.error || 'Failed to save' });
      }
    } catch (error) {
      setFieldErrors({ ...fieldErrors, [field]: 'Failed to save changes' });
    } finally {
      setSavingField(null);
    }
  };

  const renderEditableField = (
    label: string,
    icon: React.ReactNode,
    field: string,
    dbField: string,
    currentValue: any,
    displayValue: string,
    inputType: 'text' | 'number' = 'text',
    colorClass: string = 'blue',
    isUndefined: boolean = false
  ) => {
    const isEditing = editingField === field;
    const isSaving = savingField === field;
    const isSuccess = saveSuccess === field;
    const error = fieldErrors[field];

    return (
      <div className={`p-3 ${isUndefined ? 'bg-slate-50 border-slate-200' : `bg-gradient-to-r from-${colorClass}-50 to-${colorClass}-100/50 border-${colorClass}-200/50`} rounded-lg border`}>
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-2">
            {icon}
            <span className="font-medium text-slate-700 text-sm">{label}</span>
          </div>
          <div className="flex items-center gap-2">
            {!isEditing ? (
              <>
                <span className={`px-3 py-1 rounded-md text-sm font-semibold shadow-sm ${isUndefined ? 'text-slate-400 bg-white' : 'text-slate-900 bg-white'}`}>
                  {displayValue}
                </span>
                {isSuccess ? (
                  <div className="text-emerald-600 animate-in fade-in">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"/>
                    </svg>
                  </div>
                ) : (
                  <button
                    onClick={() => startEditing(field, currentValue ?? (inputType === 'number' ? 0 : ''))}
                    className="text-slate-400 hover:text-slate-600 p-1 hover:bg-white/50 rounded transition-colors"
                    title="Edit"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                    </svg>
                  </button>
                )}
              </>
            ) : (
              <div className="flex items-center gap-1">
                <input
                  type={inputType}
                  value={editValues[field] ?? currentValue}
                  onChange={(e) => setEditValues({ ...editValues, [field]: e.target.value })}
                  className="w-24 px-2 py-1 text-sm border border-slate-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                  autoFocus
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      saveField(field, dbField, editValues[field] ?? currentValue);
                    } else if (e.key === 'Escape') {
                      cancelEditing();
                    }
                  }}
                />
                <button
                  onClick={() => saveField(field, dbField, editValues[field] ?? currentValue)}
                  disabled={isSaving}
                  className="text-emerald-600 hover:text-emerald-700 p-1 hover:bg-white/50 rounded transition-colors disabled:opacity-50"
                  title="Save"
                >
                  {isSaving ? (
                    <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                    </svg>
                  ) : (
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"/>
                    </svg>
                  )}
                </button>
                <button
                  onClick={cancelEditing}
                  disabled={isSaving}
                  className="text-slate-400 hover:text-slate-600 p-1 hover:bg-white/50 rounded transition-colors disabled:opacity-50"
                  title="Cancel"
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd"/>
                  </svg>
                </button>
              </div>
            )}
          </div>
        </div>
        {error && (
          <div className="mt-2 text-xs text-red-600 bg-red-50 px-2 py-1 rounded">
            {error}
          </div>
        )}
      </div>
    );
  };

  const renderToggleField = (
    label: string,
    icon: React.ReactNode,
    field: string,
    dbField: string,
    currentValue: boolean | undefined,
    colorClass: string = 'emerald'
  ) => {
    const isSaving = savingField === field;
    const isSuccess = saveSuccess === field;

    const handleToggle = async () => {
      const newValue = currentValue === undefined ? true : !currentValue;
      setSavingField(field);

      try {
        const result = await updateFragmentMetadata(fragment.id, { [dbField]: newValue ? 1 : 0 });

        if (result.success) {
          // Refetch the fragment to get updated data
          const updatedFragment = await getFragmentById(fragment.id);
          if (updatedFragment) {
            setFragment(updatedFragment);
          }

          setSaveSuccess(field);
          setTimeout(() => setSaveSuccess(null), 2000);

          if (onUpdate) {
            onUpdate();
          }
        } else {
          setFieldErrors({ ...fieldErrors, [field]: result.error || 'Failed to save' });
        }
      } catch (error) {
        setFieldErrors({ ...fieldErrors, [field]: 'Failed to save changes' });
      } finally {
        setSavingField(null);
      }
    };

    const hasValue = currentValue !== undefined;
    const isActive = currentValue === true;

    return (
      <div className={`p-3 rounded-lg border ${
        hasValue && isActive
          ? `bg-gradient-to-r from-${colorClass}-50 to-${colorClass}-100/50 border-${colorClass}-200/50`
          : 'bg-gradient-to-r from-slate-50 to-slate-100/50 border-slate-200/50'
      }`}>
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-2">
            {icon}
            <span className="font-medium text-slate-700 text-sm">{label}</span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleToggle}
              disabled={isSaving}
              className={`px-3 py-1 rounded-md text-sm font-semibold shadow-sm flex items-center gap-1.5 transition-colors disabled:opacity-50 ${
                hasValue && isActive
                  ? colorClass === 'cyan'
                    ? 'bg-white text-cyan-700 hover:bg-cyan-50'
                    : 'bg-white text-emerald-700 hover:bg-emerald-50'
                  : hasValue
                  ? 'bg-white text-slate-600 hover:bg-slate-50'
                  : 'bg-white text-slate-400 hover:bg-slate-50'
              }`}
            >
              {isSaving ? (
                <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                </svg>
              ) : hasValue && isActive ? (
                <>
                  <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"/>
                  </svg>
                  Yes
                </>
              ) : hasValue ? (
                <>
                  <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd"/>
                  </svg>
                  No
                </>
              ) : (
                'Not detected'
              )}
            </button>
            {isSuccess && (
              <div className="text-emerald-600 animate-in fade-in">
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"/>
                </svg>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  // Calculate scale from canvas resize
  const calculateScaleFromResize = () => {
    if (!canvasFragment || !canvasFragment.originalWidth || !canvasFragment.originalHeight) {
      return null;
    }

    // Current display size (after resize)
    const displayWidth = canvasFragment.width;
    const displayHeight = canvasFragment.height;

    // Original image size
    const originalWidth = canvasFragment.originalWidth;
    const originalHeight = canvasFragment.originalHeight;

    // Calculate physical size in cm based on grid scale (pixels per cm)
    const widthInCm = displayWidth / gridScale;
    const heightInCm = displayHeight / gridScale;

    // Calculate pixels per cm for the original image
    // If the user resized to align with grid, we can calculate:
    // pixelsPerCm = (original pixels) / (physical cm)
    const pixelsPerCmWidth = originalWidth / widthInCm;
    const pixelsPerCmHeight = originalHeight / heightInCm;

    // Use average or width-based (assuming uniform scaling)
    const pixelsPerCm = (pixelsPerCmWidth + pixelsPerCmHeight) / 2;

    return {
      pixelsPerUnit: pixelsPerCm,
      unit: 'cm' as const,
      widthInCm,
      heightInCm,
    };
  };

  // Handle setting scale from resize
  const handleSetScaleFromResize = async () => {
    const calculatedScale = calculateScaleFromResize();
    if (!calculatedScale) return;

    setSavingField('scaleFromResize');

    try {
      const result = await updateFragmentMetadata(fragment.id, {
        pixels_per_unit: calculatedScale.pixelsPerUnit,
        scale_unit: calculatedScale.unit,
        scale_detection_status: 'success'
      });

      if (result.success) {
        // Refetch the fragment to get updated data
        const updatedFragment = await getFragmentById(fragment.id);
        if (updatedFragment) {
          setFragment(updatedFragment);
        }

        setSaveSuccess('scaleFromResize');
        setTimeout(() => setSaveSuccess(null), 2000);

        if (onUpdate) {
          onUpdate();
        }
      } else {
        setFieldErrors({ ...fieldErrors, scaleFromResize: result.error || 'Failed to save' });
      }
    } catch (error) {
      setFieldErrors({ ...fieldErrors, scaleFromResize: 'Failed to save scale' });
    } finally {
      setSavingField(null);
    }
  };

  const renderDropdownField = (
    label: string,
    icon: React.ReactNode,
    field: string,
    dbField: string,
    currentValue: string | undefined,
    options: readonly string[],
    colorClass: string = 'purple',
    isUndefined: boolean = false
  ) => {
    const isEditing = editingField === field;
    const isSaving = savingField === field;
    const isSuccess = saveSuccess === field;
    const error = fieldErrors[field];
    const [showDropdown, setShowDropdown] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);

    // Close dropdown when clicking outside
    useEffect(() => {
      const handleClickOutside = (event: MouseEvent) => {
        if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
          setShowDropdown(false);
        }
      };

      if (showDropdown) {
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
      }
    }, [showDropdown]);

    const handleSelect = (value: string) => {
      setEditValues({ ...editValues, [field]: value });
      setShowDropdown(false);
    };

    const handleSave = async () => {
      const valueToSave = editValues[field] ?? currentValue ?? '';
      await saveField(field, dbField, valueToSave);
    };

    return (
      <div className={`p-3 ${isUndefined ? 'bg-slate-50 border-slate-200' : `bg-gradient-to-r from-${colorClass}-50 to-${colorClass}-100/50 border-${colorClass}-200/50`} rounded-lg border`}>
        <div className="flex justify-between items-start">
          <div className="flex items-center gap-2">
            {icon}
            <span className="font-medium text-slate-700 text-sm">{label}</span>
          </div>
          <div className="flex items-center gap-2 flex-1 justify-end">
            {!isEditing ? (
              <>
                <span className={`px-3 py-1 rounded-md text-xs font-semibold shadow-sm max-w-[180px] truncate text-right ${isUndefined ? 'text-slate-400 bg-white' : 'text-slate-900 bg-white'}`} title={currentValue || 'Not detected'}>
                  {currentValue || 'Not detected'}
                </span>
                {isSuccess ? (
                  <div className="text-emerald-600 animate-in fade-in">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"/>
                    </svg>
                  </div>
                ) : (
                  <button
                    onClick={() => startEditing(field, currentValue ?? '')}
                    className="text-slate-400 hover:text-slate-600 p-1 hover:bg-white/50 rounded transition-colors flex-shrink-0"
                    title="Edit"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                    </svg>
                  </button>
                )}
              </>
            ) : (
              <div className="flex items-start gap-1 flex-1 justify-end">
                <div ref={dropdownRef} className="relative flex-1 max-w-[200px]">
                  <button
                    onClick={() => setShowDropdown(!showDropdown)}
                    className="w-full px-2 py-1 text-xs border border-slate-300 rounded bg-white focus:outline-none focus:ring-2 focus:ring-purple-500 text-left flex items-center justify-between"
                  >
                    <span className="truncate">{editValues[field] || currentValue || 'Select...'}</span>
                    <svg className="w-3 h-3 ml-1 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>
                  {showDropdown && (
                    <div className="absolute z-50 w-full mt-1 bg-white border border-slate-300 rounded-md shadow-lg max-h-60 overflow-y-auto">
                      <div
                        className="px-2 py-1.5 text-xs hover:bg-slate-100 cursor-pointer text-slate-500"
                        onClick={() => handleSelect('')}
                      >
                        Clear selection
                      </div>
                      {options.map((option) => (
                        <div
                          key={option}
                          className="px-2 py-1.5 text-xs hover:bg-purple-50 cursor-pointer"
                          onClick={() => handleSelect(option)}
                        >
                          {option}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                <button
                  onClick={handleSave}
                  disabled={isSaving}
                  className="text-emerald-600 hover:text-emerald-700 p-1 hover:bg-white/50 rounded transition-colors disabled:opacity-50 flex-shrink-0"
                  title="Save"
                >
                  {isSaving ? (
                    <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                    </svg>
                  ) : (
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"/>
                    </svg>
                  )}
                </button>
                <button
                  onClick={cancelEditing}
                  disabled={isSaving}
                  className="text-slate-400 hover:text-slate-600 p-1 hover:bg-white/50 rounded transition-colors disabled:opacity-50 flex-shrink-0"
                  title="Cancel"
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd"/>
                  </svg>
                </button>
              </div>
            )}
          </div>
        </div>
        {error && (
          <div className="mt-2 text-xs text-red-600 bg-red-50 px-2 py-1 rounded">
            {error}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="fixed left-1/2 top-20 -translate-x-1/2 z-50 w-96 max-w-[calc(100vw-2rem)] bg-white rounded-xl shadow-2xl border border-slate-200 overflow-hidden animate-in fade-in slide-in-from-top-4 duration-200">
      {/* Header */}
      <div className="bg-gradient-to-br from-slate-700 via-slate-800 to-slate-900 text-white p-5 flex justify-between items-start">
        <div className="flex-1 flex items-start gap-3">
          <div className="bg-white/10 rounded-lg p-2 backdrop-blur-sm">
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z"/>
              <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd"/>
            </svg>
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-base mb-1">Fragment Metadata</h3>
            <p className="text-xs text-slate-300 truncate" title={fragment.name}>
              {fragment.name}
            </p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="ml-2 text-slate-300 hover:text-white hover:bg-white/10 rounded-lg w-8 h-8 flex items-center justify-center transition-all duration-200"
          aria-label="Close"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Content */}
      <div className="p-5 max-h-[calc(100vh-16rem)] overflow-y-auto">
        <div className="space-y-3">
          {/* Line Count */}
          {renderEditableField(
            'Line Count',
            <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>,
            'lineCount',
            'line_count',
            metadata?.lineCount,
            metadata?.lineCount !== undefined ? String(metadata.lineCount) : 'Not detected',
            'number',
            'blue',
            metadata?.lineCount === undefined
          )}

          {/* Script Type */}
          {renderDropdownField(
            'Script Type',
            <svg className="w-4 h-4 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
            </svg>,
            'script',
            'script_type',
            metadata?.script,
            SCRIPT_TYPES,
            'purple',
            metadata?.script === undefined
          )}

          {/* Edge Piece */}
          {renderToggleField(
            'Edge Piece',
            <svg className={`w-4 h-4 ${metadata?.isEdgePiece ? 'text-emerald-600' : 'text-slate-400'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>,
            'isEdgePiece',
            'edge_piece',
            metadata?.isEdgePiece,
            'emerald'
          )}

          {/* Top Edge */}
          {renderToggleField(
            'Top Edge',
            <svg className={`w-4 h-4 ${metadata?.hasTopEdge ? 'text-emerald-600' : 'text-slate-400'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
            </svg>,
            'hasTopEdge',
            'has_top_edge',
            metadata?.hasTopEdge,
            'emerald'
          )}

          {/* Bottom Edge */}
          {renderToggleField(
            'Bottom Edge',
            <svg className={`w-4 h-4 ${metadata?.hasBottomEdge ? 'text-emerald-600' : 'text-slate-400'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>,
            'hasBottomEdge',
            'has_bottom_edge',
            metadata?.hasBottomEdge,
            'emerald'
          )}

          {/* Left Edge */}
          {renderToggleField(
            'Left Edge',
            <svg className={`w-4 h-4 ${metadata?.hasLeftEdge ? 'text-emerald-600' : 'text-slate-400'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>,
            'hasLeftEdge',
            'has_left_edge',
            metadata?.hasLeftEdge,
            'emerald'
          )}

          {/* Right Edge */}
          {renderToggleField(
            'Right Edge',
            <svg className={`w-4 h-4 ${metadata?.hasRightEdge ? 'text-emerald-600' : 'text-slate-400'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>,
            'hasRightEdge',
            'has_right_edge',
            metadata?.hasRightEdge,
            'emerald'
          )}

          {/* Circle Detection */}
          {renderToggleField(
            'Has Circle',
            <svg className={`w-4 h-4 ${metadata?.hasCircle ? 'text-cyan-600' : 'text-slate-400'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="9" strokeWidth={2} />
            </svg>,
            'hasCircle',
            'has_circle',
            metadata?.hasCircle,
            'cyan'
          )}

          {/* Scale Information */}
          {metadata?.scale ? (
            <div className="p-3 bg-gradient-to-r from-amber-50 to-amber-100/50 rounded-lg border border-amber-200/50">
              <div className="flex justify-between items-start mb-2">
                <div className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" />
                  </svg>
                  <span className="font-medium text-slate-700 text-sm">Scale Information</span>
                </div>
                <span className={`text-xs px-2 py-0.5 rounded ${
                  metadata.scale.detectionStatus === 'success' ? 'bg-emerald-100 text-emerald-700' : 'bg-red-100 text-red-700'
                }`}>
                  {metadata.scale.detectionStatus === 'success' ? 'Auto-detected' : 'Detection failed'}
                </span>
              </div>
              {renderEditableField(
                'Pixels per Unit',
                <span className="text-xs text-slate-500">Value:</span>,
                'pixelsPerUnit',
                'pixels_per_unit',
                metadata.scale.pixelsPerUnit,
                `${metadata.scale.pixelsPerUnit.toFixed(1)} px/${metadata.scale.unit}`,
                'number',
                'amber'
              )}
              {/* Set Scale from Resize Button */}
              {canvasFragment?.hasBeenResized && canvasFragment.originalWidth && (
                <div className="mt-2">
                  <button
                    onClick={handleSetScaleFromResize}
                    disabled={savingField === 'scaleFromResize'}
                    className="w-full px-3 py-2 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white rounded-md text-sm font-medium shadow-sm flex items-center justify-center gap-2 transition-all disabled:opacity-50"
                  >
                    {savingField === 'scaleFromResize' ? (
                      <>
                        <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                        </svg>
                        Setting Scale...
                      </>
                    ) : saveSuccess === 'scaleFromResize' ? (
                      <>
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"/>
                        </svg>
                        Scale Set!
                      </>
                    ) : (
                      <>
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                        </svg>
                        Set Scale from Resize
                      </>
                    )}
                  </button>
                  {(() => {
                    const calc = calculateScaleFromResize();
                    return calc ? (
                      <p className="text-xs text-slate-500 mt-1 text-center">
                        Will set to {calc.pixelsPerUnit.toFixed(1)} px/cm
                        <span className="text-slate-400"> ({calc.widthInCm.toFixed(1)} × {calc.heightInCm.toFixed(1)} cm)</span>
                      </p>
                    ) : null;
                  })()}
                </div>
              )}
            </div>
          ) : (
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
              <div className="flex justify-between items-center mb-2">
                <div className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" />
                  </svg>
                  <span className="font-medium text-slate-700 text-sm">Scale Information</span>
                </div>
                <span className="text-slate-400 text-sm">No ruler detected</span>
              </div>
              {/* Set Scale from Resize Button - also show when no scale exists */}
              {canvasFragment?.hasBeenResized && canvasFragment.originalWidth && (
                <div className="mt-2">
                  <button
                    onClick={handleSetScaleFromResize}
                    disabled={savingField === 'scaleFromResize'}
                    className="w-full px-3 py-2 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white rounded-md text-sm font-medium shadow-sm flex items-center justify-center gap-2 transition-all disabled:opacity-50"
                  >
                    {savingField === 'scaleFromResize' ? (
                      <>
                        <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                        </svg>
                        Setting Scale...
                      </>
                    ) : saveSuccess === 'scaleFromResize' ? (
                      <>
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"/>
                        </svg>
                        Scale Set!
                      </>
                    ) : (
                      <>
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                        </svg>
                        Set Scale from Resize
                      </>
                    )}
                  </button>
                  {(() => {
                    const calc = calculateScaleFromResize();
                    return calc ? (
                      <p className="text-xs text-slate-500 mt-1 text-center">
                        Will set to {calc.pixelsPerUnit.toFixed(1)} px/cm
                        <span className="text-slate-400"> ({calc.widthInCm.toFixed(1)} × {calc.heightInCm.toFixed(1)} cm)</span>
                      </p>
                    ) : null;
                  })()}
                </div>
              )}
            </div>
          )}

          {/* Segmentation Status */}
          {fragment.hasSegmentation ? (
            <div className="p-3 bg-gradient-to-r from-indigo-50 to-indigo-100/50 rounded-lg border border-indigo-200/50">
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM14 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1H5a1 1 0 01-1-1v-3zM14 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1h-4a1 1 0 01-1-1v-3z" />
                  </svg>
                  <span className="font-medium text-slate-700 text-sm">Segmentation</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-indigo-700 bg-white px-3 py-1 rounded-md text-sm font-semibold shadow-sm">
                    Segmented
                  </span>
                  <svg className="w-3.5 h-3.5 text-emerald-600" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"/>
                  </svg>
                </div>
              </div>
            </div>
          ) : (
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM14 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1H5a1 1 0 01-1-1v-3zM14 16a1 1 0 011-1h4a1 1 0 011 1v3a1 1 0 01-1 1h-4a1 1 0 01-1-1v-3z" />
                  </svg>
                  <span className="font-medium text-slate-700 text-sm">Segmentation</span>
                </div>
                <span className="text-slate-400 text-sm">Not segmented</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Footer note */}
      <div className="bg-slate-50 px-5 py-3 text-xs text-slate-500 border-t border-slate-200 flex items-center gap-2">
        <svg className="w-3.5 h-3.5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <span>Click field values to edit • Press ESC to close</span>
      </div>
    </div>
  );
};

export default FragmentMetadata;
