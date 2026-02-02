import React, { useState, useEffect } from 'react';
import { ManuscriptFragment } from '../types/fragment';
import { updateFragmentMetadata, getFragmentById } from '../services/fragment-service';
import {
  validateLineCount,
  validatePixelsPerUnit,
  validateScriptType,
  validateScaleUnit,
} from '../utils/metadataValidation';

interface FragmentMetadataProps {
  fragment: ManuscriptFragment;
  onClose: () => void;
  onUpdate?: () => void; // Callback to refresh fragment data after edit
}

const FragmentMetadata: React.FC<FragmentMetadataProps> = ({ fragment: initialFragment, onClose, onUpdate }) => {
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

    try {
      const result = await updateFragmentMetadata(fragment.id, { [dbField]: value });

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
          {renderEditableField(
            'Script Type',
            <svg className="w-4 h-4 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
            </svg>,
            'script',
            'script_type',
            metadata?.script,
            metadata?.script ?? 'Not detected',
            'text',
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
            </div>
          ) : (
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" />
                  </svg>
                  <span className="font-medium text-slate-700 text-sm">Scale Information</span>
                </div>
                <span className="text-slate-400 text-sm">No ruler detected</span>
              </div>
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
