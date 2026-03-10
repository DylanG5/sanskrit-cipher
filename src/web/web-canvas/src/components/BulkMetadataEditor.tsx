import React, { useState } from 'react';
import { bulkUpdateFragmentMetadata } from '../services/fragment-service';
import {
  validateLineCount,
  validatePixelsPerUnit,
  validateScriptType,
  validateScaleUnit,
} from '../utils/metadataValidation';
import { SCRIPT_TYPES, getScriptTypeDB } from '../types/constants';
import { CustomFilterDefinition } from '../types/customFilters';

interface BulkMetadataEditorProps {
  fragmentIds: string[];
  onClose: () => void;
  onUpdate?: () => void;
  customFilters: CustomFilterDefinition[];
}

interface FieldState {
  enabled: boolean;
  value: unknown;
}

const BulkMetadataEditor: React.FC<BulkMetadataEditorProps> = ({
  fragmentIds,
  onClose,
  onUpdate,
  customFilters,
}) => {
  // Which fields are enabled (checked) and their values
  const [fields, setFields] = useState<Record<string, FieldState>>({
    script_type: { enabled: false, value: '' },
    line_count: { enabled: false, value: '' },
    edge_piece: { enabled: false, value: false },
    has_top_edge: { enabled: false, value: false },
    has_bottom_edge: { enabled: false, value: false },
    has_left_edge: { enabled: false, value: false },
    has_right_edge: { enabled: false, value: false },
    has_circle: { enabled: false, value: false },
    scale_unit: { enabled: false, value: 'cm' },
    pixels_per_unit: { enabled: false, value: '' },
    ...Object.fromEntries(
      customFilters.map((f) => [f.filterKey, { enabled: false, value: '' }])
    ),
  });

  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isSaving, setIsSaving] = useState(false);
  const [feedback, setFeedback] = useState<{ type: 'success' | 'error'; message: string } | null>(null);

  const toggleField = (key: string) => {
    setFields((prev) => ({
      ...prev,
      [key]: { ...prev[key], enabled: !prev[key].enabled },
    }));
    setErrors((prev) => ({ ...prev, [key]: '' }));
  };

  const setFieldValue = (key: string, value: unknown) => {
    setFields((prev) => ({
      ...prev,
      [key]: { ...prev[key], value },
    }));
    setErrors((prev) => ({ ...prev, [key]: '' }));
  };

  const handleApply = async () => {
    const metadata: Record<string, unknown> = {};
    const newErrors: Record<string, string> = {};

    for (const [key, field] of Object.entries(fields)) {
      if (!field.enabled) continue;

      // Validate
      if (key === 'line_count') {
        const val = field.value === '' ? null : Number(field.value);
        if (val !== null) {
          const result = validateLineCount(val);
          if (!result.valid) {
            newErrors[key] = result.error || 'Invalid';
            continue;
          }
        }
        metadata[key] = val;
      } else if (key === 'script_type') {
        const val = String(field.value || '');
        if (val) {
          const result = validateScriptType(val);
          if (!result.valid) {
            newErrors[key] = result.error || 'Invalid';
            continue;
          }
          metadata[key] = getScriptTypeDB(val) || val;
        } else {
          metadata[key] = null;
        }
      } else if (key === 'pixels_per_unit') {
        const val = field.value === '' ? null : Number(field.value);
        if (val !== null) {
          const result = validatePixelsPerUnit(val);
          if (!result.valid) {
            newErrors[key] = result.error || 'Invalid';
            continue;
          }
        }
        metadata[key] = val;
      } else if (key === 'scale_unit') {
        const val = String(field.value);
        const result = validateScaleUnit(val);
        if (!result.valid) {
          newErrors[key] = result.error || 'Invalid';
          continue;
        }
        metadata[key] = val;
      } else if (
        ['edge_piece', 'has_top_edge', 'has_bottom_edge', 'has_left_edge', 'has_right_edge', 'has_circle'].includes(key)
      ) {
        metadata[key] = field.value ? 1 : 0;
      } else {
        // Custom filter
        metadata[key] = field.value === '' ? null : field.value;
      }
    }

    setErrors(newErrors);
    if (Object.keys(newErrors).length > 0) return;
    if (Object.keys(metadata).length === 0) {
      setFeedback({ type: 'error', message: 'No fields selected to update' });
      return;
    }

    setIsSaving(true);
    setFeedback(null);

    try {
      const result = await bulkUpdateFragmentMetadata(fragmentIds, metadata);
      if (result.success) {
        setFeedback({ type: 'success', message: `Updated ${fragmentIds.length} fragments` });
        onUpdate?.();
        setTimeout(() => onClose(), 1200);
      } else {
        setFeedback({ type: 'error', message: result.error || 'Failed to update' });
      }
    } catch {
      setFeedback({ type: 'error', message: 'An unexpected error occurred' });
    } finally {
      setIsSaving(false);
    }
  };

  const renderCheckbox = (key: string) => (
    <input
      type="checkbox"
      checked={fields[key]?.enabled ?? false}
      onChange={() => toggleField(key)}
      className="w-4 h-4 rounded border-slate-300 text-orange-600 focus:ring-orange-500 cursor-pointer"
    />
  );

  const renderBoolField = (key: string, label: string) => (
    <div className="flex items-center gap-3 p-2.5 rounded-lg border border-slate-200 bg-slate-50">
      {renderCheckbox(key)}
      <span className="text-sm font-medium text-slate-700 flex-1">{label}</span>
      <select
        value={fields[key]?.value ? 'true' : 'false'}
        onChange={(e) => setFieldValue(key, e.target.value === 'true')}
        disabled={!fields[key]?.enabled}
        className="px-2 py-1 text-sm border border-slate-300 rounded bg-white disabled:opacity-40"
      >
        <option value="true">Yes</option>
        <option value="false">No</option>
      </select>
    </div>
  );

  return (
    <div className="fixed left-1/2 top-16 -translate-x-1/2 z-50 w-[28rem] max-w-[calc(100vw-2rem)] max-h-[calc(100vh-5rem)] bg-white rounded-xl shadow-2xl border border-slate-200 overflow-hidden animate-in fade-in slide-in-from-top-4 duration-200 flex flex-col">
      {/* Header */}
      <div className="bg-gradient-to-br from-slate-700 via-slate-800 to-slate-900 text-white p-5 flex justify-between items-start shrink-0">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-orange-500/20 rounded-lg">
            <svg className="w-5 h-5 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
            </svg>
          </div>
          <div>
            <h2 className="text-lg font-bold">Edit {fragmentIds.length} Fragments</h2>
            <p className="text-xs text-slate-400 mt-0.5">Check fields to include in the update</p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="text-slate-400 hover:text-white transition-colors p-1 rounded-lg hover:bg-white/10"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Body */}
      <div className="p-4 space-y-3 overflow-y-auto flex-1">
        {/* Script Type */}
        <div className="flex items-center gap-3 p-2.5 rounded-lg border border-slate-200 bg-slate-50">
          {renderCheckbox('script_type')}
          <span className="text-sm font-medium text-slate-700 w-24">Script Type</span>
          <select
            value={String(fields.script_type?.value ?? '')}
            onChange={(e) => setFieldValue('script_type', e.target.value)}
            disabled={!fields.script_type?.enabled}
            className="flex-1 px-2 py-1 text-sm border border-slate-300 rounded bg-white disabled:opacity-40"
          >
            <option value="">— Clear —</option>
            {SCRIPT_TYPES.map((st) => (
              <option key={st} value={st}>{st}</option>
            ))}
          </select>
          {errors.script_type && <span className="text-xs text-red-500">{errors.script_type}</span>}
        </div>

        {/* Line Count */}
        <div className="flex items-center gap-3 p-2.5 rounded-lg border border-slate-200 bg-slate-50">
          {renderCheckbox('line_count')}
          <span className="text-sm font-medium text-slate-700 w-24">Line Count</span>
          <input
            type="number"
            value={String(fields.line_count?.value ?? '')}
            onChange={(e) => setFieldValue('line_count', e.target.value)}
            disabled={!fields.line_count?.enabled}
            placeholder="0-100"
            min={0}
            max={100}
            className="flex-1 px-2 py-1 text-sm border border-slate-300 rounded bg-white disabled:opacity-40"
          />
          {errors.line_count && <span className="text-xs text-red-500">{errors.line_count}</span>}
        </div>

        {/* Edge flags */}
        <div className="space-y-1.5">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider px-1">Edge Flags</p>
          {renderBoolField('edge_piece', 'Edge Piece')}
          {renderBoolField('has_top_edge', 'Has Top Edge')}
          {renderBoolField('has_bottom_edge', 'Has Bottom Edge')}
          {renderBoolField('has_left_edge', 'Has Left Edge')}
          {renderBoolField('has_right_edge', 'Has Right Edge')}
          {renderBoolField('has_circle', 'Has Circle')}
        </div>

        {/* Scale */}
        <div className="space-y-1.5">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider px-1">Scale</p>
          <div className="flex items-center gap-3 p-2.5 rounded-lg border border-slate-200 bg-slate-50">
            {renderCheckbox('scale_unit')}
            <span className="text-sm font-medium text-slate-700 w-24">Scale Unit</span>
            <select
              value={String(fields.scale_unit?.value ?? 'cm')}
              onChange={(e) => setFieldValue('scale_unit', e.target.value)}
              disabled={!fields.scale_unit?.enabled}
              className="flex-1 px-2 py-1 text-sm border border-slate-300 rounded bg-white disabled:opacity-40"
            >
              <option value="cm">cm</option>
              <option value="mm">mm</option>
            </select>
            {errors.scale_unit && <span className="text-xs text-red-500">{errors.scale_unit}</span>}
          </div>
          <div className="flex items-center gap-3 p-2.5 rounded-lg border border-slate-200 bg-slate-50">
            {renderCheckbox('pixels_per_unit')}
            <span className="text-sm font-medium text-slate-700 w-24">Px/Unit</span>
            <input
              type="number"
              value={String(fields.pixels_per_unit?.value ?? '')}
              onChange={(e) => setFieldValue('pixels_per_unit', e.target.value)}
              disabled={!fields.pixels_per_unit?.enabled}
              placeholder="e.g. 25"
              className="flex-1 px-2 py-1 text-sm border border-slate-300 rounded bg-white disabled:opacity-40"
            />
            {errors.pixels_per_unit && <span className="text-xs text-red-500">{errors.pixels_per_unit}</span>}
          </div>
        </div>

        {/* Custom Filters */}
        {customFilters.length > 0 && (
          <div className="space-y-1.5">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider px-1">Custom Filters</p>
            {customFilters.map((cf) => (
              <div key={cf.filterKey} className="flex items-center gap-3 p-2.5 rounded-lg border border-slate-200 bg-slate-50">
                {renderCheckbox(cf.filterKey)}
                <span className="text-sm font-medium text-slate-700 w-24 truncate" title={cf.label}>{cf.label}</span>
                {cf.type === 'multiselect' && cf.options ? (
                  <select
                    value={String(fields[cf.filterKey]?.value ?? '')}
                    onChange={(e) => setFieldValue(cf.filterKey, e.target.value)}
                    disabled={!fields[cf.filterKey]?.enabled}
                    className="flex-1 px-2 py-1 text-sm border border-slate-300 rounded bg-white disabled:opacity-40"
                  >
                    <option value="">— Clear —</option>
                    {cf.options.map((opt) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="text"
                    value={String(fields[cf.filterKey]?.value ?? '')}
                    onChange={(e) => setFieldValue(cf.filterKey, e.target.value)}
                    disabled={!fields[cf.filterKey]?.enabled}
                    className="flex-1 px-2 py-1 text-sm border border-slate-300 rounded bg-white disabled:opacity-40"
                  />
                )}
                {errors[cf.filterKey] && <span className="text-xs text-red-500">{errors[cf.filterKey]}</span>}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-slate-200 flex items-center justify-between shrink-0 bg-slate-50">
        {feedback ? (
          <span className={`text-sm font-medium ${feedback.type === 'success' ? 'text-emerald-600' : 'text-red-600'}`}>
            {feedback.message}
          </span>
        ) : (
          <span className="text-xs text-slate-400">
            {Object.values(fields).filter((f) => f.enabled).length} field(s) selected
          </span>
        )}
        <div className="flex gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-slate-600 bg-white border border-slate-300 rounded-lg hover:bg-slate-50"
          >
            Cancel
          </button>
          <button
            onClick={handleApply}
            disabled={isSaving || Object.values(fields).every((f) => !f.enabled)}
            className="px-4 py-2 text-sm font-bold text-white rounded-lg shadow-md disabled:opacity-40"
            style={{ background: 'linear-gradient(135deg, #ea580c 0%, #c2410c 100%)' }}
          >
            {isSaving ? 'Applying...' : 'Apply'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default BulkMetadataEditor;
