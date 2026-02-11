import React, { useState, useRef, useCallback } from 'react';
import { FragmentFilters, DEFAULT_FILTERS } from '../types/filters';
import { CustomFilterDefinition } from '../types/customFilters';

interface FilterPanelProps {
  filters: FragmentFilters;
  onFiltersChange: (filters: FragmentFilters) => void;
  availableScripts: string[];
  matchCount: number;
  totalCount: number;
  isOpen: boolean;
  onToggle: () => void;
  width: number;
  onWidthChange: (width: number) => void;
  customFilters: CustomFilterDefinition[];
  onCreateCustomFilter: (payload: {
    label: string;
    type: 'dropdown' | 'text';
    options?: string[];
  }) => Promise<CustomFilterDefinition | null>;
  onDeleteCustomFilter: (id: number) => Promise<boolean>;
  onUpdateCustomFilterOptions: (id: number, options: string[]) => Promise<CustomFilterDefinition | null>;
}

const FilterPanel: React.FC<FilterPanelProps> = ({
  filters,
  onFiltersChange,
  availableScripts,
  matchCount,
  totalCount,
  isOpen,
  onToggle,
  width,
  onWidthChange,
  customFilters,
  onCreateCustomFilter,
  onDeleteCustomFilter,
  onUpdateCustomFilterOptions,
}) => {
  const [localFilters, setLocalFilters] = useState<FragmentFilters>(filters);
  const [isResizing, setIsResizing] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);
  const [isAddingCustom, setIsAddingCustom] = useState(false);
  const [newFilterLabel, setNewFilterLabel] = useState('');
  const [newFilterType, setNewFilterType] = useState<'dropdown' | 'text'>('dropdown');
  const [newFilterOptions, setNewFilterOptions] = useState('');
  const [newFilterError, setNewFilterError] = useState<string | null>(null);
  const [isCreatingCustom, setIsCreatingCustom] = useState(false);
  const [editingOptionsId, setEditingOptionsId] = useState<number | null>(null);
  const [optionsDraft, setOptionsDraft] = useState('');
  const [optionsError, setOptionsError] = useState<string | null>(null);
  const [isSavingOptions, setIsSavingOptions] = useState(false);

  React.useEffect(() => {
    setLocalFilters(filters);
  }, [filters]);

  React.useEffect(() => {
    if (customFilters.length === 0) {
      setLocalFilters((prev) => ({ ...prev, custom: {} }));
      return;
    }
    setLocalFilters((prev) => {
      const validKeys = new Set(customFilters.map(f => f.filterKey));
      const cleanedCustom = Object.fromEntries(
        Object.entries(prev.custom || {}).filter(([k]) => validKeys.has(k))
      );
      // Add new filter keys that don't exist yet
      customFilters.forEach(filter => {
        if (!(filter.filterKey in cleanedCustom)) {
          cleanedCustom[filter.filterKey] = undefined;
        }
      });
      return { ...prev, custom: cleanedCustom };
    });
  }, [customFilters]);

  const handleLineCountMinChange = (value: string) => {
    const num = value === '' ? undefined : parseInt(value);
    setLocalFilters({ ...localFilters, lineCountMin: num });
  };

  const handleLineCountMaxChange = (value: string) => {
    const num = value === '' ? undefined : parseInt(value);
    setLocalFilters({ ...localFilters, lineCountMax: num });
  };

  const handleScriptToggle = (script: string) => {
    const scripts = localFilters.scripts.includes(script)
      ? localFilters.scripts.filter(s => s !== script)
      : [...localFilters.scripts, script];
    setLocalFilters({ ...localFilters, scripts });
  };

  const handleEdgePieceChange = (value: boolean | null) => {
    setLocalFilters({ ...localFilters, isEdgePiece: value });
  };

  const handleTopEdgeChange = (value: boolean | null) => {
    setLocalFilters({ ...localFilters, hasTopEdge: value });
  };

  const handleBottomEdgeChange = (value: boolean | null) => {
    setLocalFilters({ ...localFilters, hasBottomEdge: value });
  };

  const handleLeftEdgeChange = (value: boolean | null) => {
    setLocalFilters({ ...localFilters, hasLeftEdge: value });
  };

  const handleRightEdgeChange = (value: boolean | null) => {
    setLocalFilters({ ...localFilters, hasRightEdge: value });
  };

  const handleCircleChange = (value: boolean | null) => {
    setLocalFilters({ ...localFilters, hasCircle: value });
  };

  const handleCustomValueChange = (key: string, value: string) => {
    const nextValue = value.trim() === '' ? undefined : value;
    setLocalFilters({
      ...localFilters,
      custom: {
        ...(localFilters.custom || {}),
        [key]: nextValue,
      },
    });
  };

  const handleApply = () => {
    onFiltersChange(localFilters);
  };

  const handleReset = () => {
    setLocalFilters(DEFAULT_FILTERS);
    onFiltersChange(DEFAULT_FILTERS);
  };

  const hasCustomFilters = Object.values(localFilters.custom || {}).some(
    (value) => value !== undefined && value !== null && value !== ''
  );

  const hasActiveFilters =
    localFilters.lineCountMin !== undefined ||
    localFilters.lineCountMax !== undefined ||
    localFilters.scripts.length > 0 ||
    localFilters.isEdgePiece !== null ||
    localFilters.hasTopEdge !== null ||
    localFilters.hasBottomEdge !== null ||
    localFilters.hasLeftEdge !== null ||
    localFilters.hasRightEdge !== null ||
    localFilters.hasCircle !== null ||
    hasCustomFilters;

  const parseOptions = (input: string): string[] =>
    input
      .split(/[\n,]/g)
      .map((option) => option.trim())
      .filter((option) => option.length > 0);

  const handleCreateCustom = async () => {
    setNewFilterError(null);
    const label = newFilterLabel.trim();
    if (!label) {
      setNewFilterError('Enter a filter name.');
      return;
    }

    let options: string[] | undefined;
    if (newFilterType === 'dropdown') {
      options = parseOptions(newFilterOptions);
      if (!options.length) {
        setNewFilterError('Add at least one option.');
        return;
      }
    }

    setIsCreatingCustom(true);
    const created = await onCreateCustomFilter({
      label,
      type: newFilterType,
      options,
    });
    setIsCreatingCustom(false);

    if (!created) {
      setNewFilterError('Failed to create filter.');
      return;
    }

    setNewFilterLabel('');
    setNewFilterOptions('');
    setNewFilterType('dropdown');
    setIsAddingCustom(false);
  };

  const handleDeleteCustomFilter = async (filter: CustomFilterDefinition) => {
    const confirmed = window.confirm(`Delete custom filter "${filter.label}"?`);
    if (!confirmed) {
      return;
    }
    await onDeleteCustomFilter(filter.id);
  };

  const handleStartEditOptions = (filter: CustomFilterDefinition) => {
    setEditingOptionsId(filter.id);
    setOptionsDraft((filter.options || []).join(', '));
    setOptionsError(null);
  };

  const handleCancelEditOptions = () => {
    setEditingOptionsId(null);
    setOptionsDraft('');
    setOptionsError(null);
  };

  const handleSaveOptions = async (filter: CustomFilterDefinition) => {
    const options = parseOptions(optionsDraft);
    if (!options.length) {
      setOptionsError('Add at least one option.');
      return;
    }

    // Check if current filter value would be cleared
    const currentValue = localFilters.custom?.[filter.filterKey];
    if (currentValue && !options.includes(currentValue)) {
      const confirmed = window.confirm(
        `The current selection "${currentValue}" will be removed from the options. This will clear this value from any fragments using it. Continue?`
      );
      if (!confirmed) return;
    }

    // Check if any of the old options are being removed
    const oldOptions = filter.options || [];
    const removedOptions = oldOptions.filter(opt => !options.includes(opt));
    if (removedOptions.length > 0) {
      const confirmed = window.confirm(
        `Removing option(s): ${removedOptions.join(', ')}\n\nThis will clear these values from all fragments. Continue?`
      );
      if (!confirmed) return;
    }

    setIsSavingOptions(true);
    const updated = await onUpdateCustomFilterOptions(filter.id, options);
    setIsSavingOptions(false);
    if (!updated) {
      setOptionsError('Failed to update options.');
      return;
    }
    setEditingOptionsId(null);
    setOptionsDraft('');
    setOptionsError(null);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  };

  const handleMouseMove = useCallback((e: MouseEvent) => {
    const newWidth = window.innerWidth - e.clientX;
    // Constrain width between 280px and 450px
    if (newWidth >= 280 && newWidth <= 450) {
      onWidthChange(newWidth);
    }
  }, [onWidthChange]);

  const handleMouseUp = useCallback(() => {
    setIsResizing(false);
  }, []);

  React.useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isResizing, handleMouseMove, handleMouseUp]);

  return (
    <>
      {/* Toggle button when closed */}
      {!isOpen && (
        <button
          onClick={onToggle}
          className="fixed right-0 top-1/2 -translate-y-1/2 text-white px-3 py-6 rounded-l-xl shadow-xl transition-all duration-300 z-30 group font-body"
          style={{
            background: 'linear-gradient(270deg, #292524 0%, #44403c 100%)'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'linear-gradient(270deg, #d97706 0%, #b45309 100%)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'linear-gradient(270deg, #292524 0%, #44403c 100%)';
          }}
          title="Open filters"
        >
          <div className="flex flex-col items-center gap-2.5">
            <svg className="w-5 h-5 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
            </svg>
            <span className="text-xs font-bold font-body" style={{ writingMode: 'vertical-rl' }}>Filters</span>
            {hasActiveFilters && (
              <div className="w-2 h-2 rounded-full animate-pulse" style={{ background: '#ea580c' }}></div>
            )}
          </div>
        </button>
      )}

      {/* Filter Panel */}
      {isOpen && (
        <div
          ref={panelRef}
          style={{
            width: `${width}px`,
            background: 'linear-gradient(180deg, #fafaf9 0%, #f5f5f4 100%)',
            borderLeft: '1px solid rgba(120, 113, 108, 0.2)'
          }}
          className="overflow-y-auto shadow-lg flex flex-col flex-shrink-0 relative"
        >
          {/* Header */}
          <div className="p-4 text-white sticky top-0 z-10 shadow-md font-body" style={{
            background: 'linear-gradient(90deg, #292524 0%, #1c1917 100%)'
          }}>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2.5">
                <svg className="w-5 h-5" style={{ color: '#d97706' }} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
                </svg>
                <h2 className="text-lg font-bold font-body">Filter Fragments</h2>
              </div>
              <button
                onClick={onToggle}
                className="text-slate-300 hover:text-white hover:bg-white/10 rounded-lg p-1.5 transition-all duration-200"
                title="Close filters"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Info text */}
            <div className="mb-3 text-xs text-slate-300 bg-white/5 rounded-lg px-3 py-2 border border-white/10">
              <span className="font-semibold">💡 Tip:</span> You can apply multiple filters together - they work as AND conditions
            </div>

            {/* Match count */}
            <div className="bg-white/10 rounded-lg px-3 py-2 backdrop-blur-sm">
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-300">Showing</span>
                <span className="font-semibold text-white">
                  {matchCount} / {totalCount} fragments
                </span>
              </div>
            </div>

            {/* Active filters summary */}
            {hasActiveFilters && (
              <div className="mt-3 bg-blue-500/20 border border-blue-400/30 rounded-lg px-3 py-2">
                <div className="flex items-start gap-2">
                  <svg className="w-4 h-4 text-blue-300 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <div className="flex-1 text-xs">
                    <div className="font-semibold text-blue-200 mb-1">Active Filters:</div>
                    <div className="space-y-0.5 text-blue-100">
                      {filters.lineCountMin !== undefined && (
                        <div>• Line count min: {filters.lineCountMin}</div>
                      )}
                      {filters.lineCountMax !== undefined && (
                        <div>• Line count max: {filters.lineCountMax}</div>
                      )}
                      {filters.scripts.length > 0 && (
                        <div>• Scripts: {filters.scripts.join(', ')}</div>
                      )}
                      {filters.isEdgePiece === true && (
                        <div>• Edge pieces only</div>
                      )}
                      {filters.isEdgePiece === false && (
                        <div>• Non-edge pieces only</div>
                      )}
                      {filters.hasTopEdge === true && (
                        <div>• Top edge only</div>
                      )}
                      {filters.hasTopEdge === false && (
                        <div>• No top edge only</div>
                      )}
                      {filters.hasBottomEdge === true && (
                        <div>• Bottom edge only</div>
                      )}
                      {filters.hasBottomEdge === false && (
                        <div>• No bottom edge only</div>
                      )}
                      {filters.hasLeftEdge === true && (
                        <div>• Left edge only</div>
                      )}
                      {filters.hasLeftEdge === false && (
                        <div>• No left edge only</div>
                      )}
                      {filters.hasRightEdge === true && (
                        <div>• Right edge only</div>
                      )}
                      {filters.hasRightEdge === false && (
                        <div>• No right edge only</div>
                      )}
                      {filters.hasCircle === true && (
                        <div>• Has circle only</div>
                      )}
                      {filters.hasCircle === false && (
                        <div>• No circle only</div>
                      )}
                      {customFilters.map((filter) => {
                        const value = filters.custom?.[filter.filterKey];
                        if (!value) {
                          return null;
                        }
                        return (
                          <div key={filter.filterKey}>
                            • {filter.label}: {value}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Filter Controls */}
          <div className="flex-1 p-4 space-y-6">
            {/* Line Count Filter */}
            <div className={`bg-white rounded-lg p-4 shadow-sm border-2 transition-all ${
              (localFilters.lineCountMin !== undefined || localFilters.lineCountMax !== undefined)
                ? 'border-blue-400 ring-2 ring-blue-100'
                : 'border-slate-200'
            }`}>
              <div className="flex items-center gap-2 mb-3">
                <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
                <h3 className="font-semibold text-slate-800 text-sm">Line Count</h3>
                {(localFilters.lineCountMin !== undefined || localFilters.lineCountMax !== undefined) && (
                  <span className="ml-auto text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full font-semibold">Active</span>
                )}
              </div>
              <div className="flex items-center gap-3">
                <div className="flex-1">
                  <label className="block text-xs text-slate-600 mb-1">Min</label>
                  <input
                    type="number"
                    min="0"
                    value={localFilters.lineCountMin ?? ''}
                    onChange={(e) => handleLineCountMinChange(e.target.value)}
                    placeholder="Any"
                    className="w-full px-3 py-2 border border-slate-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                  />
                </div>
                <div className="text-slate-400 mt-5">—</div>
                <div className="flex-1">
                  <label className="block text-xs text-slate-600 mb-1">Max</label>
                  <input
                    type="number"
                    min="0"
                    value={localFilters.lineCountMax ?? ''}
                    onChange={(e) => handleLineCountMaxChange(e.target.value)}
                    placeholder="Any"
                    className="w-full px-3 py-2 border border-slate-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                  />
                </div>
              </div>
              {(localFilters.lineCountMin !== undefined || localFilters.lineCountMax !== undefined) && (
                <p className="mt-2 text-xs text-slate-500">
                  {localFilters.lineCountMin ?? '0'} - {localFilters.lineCountMax ?? '∞'} lines
                </p>
              )}
            </div>

            {/* Script Type Filter */}
            <div className={`bg-white rounded-lg p-4 shadow-sm border-2 transition-all ${
              localFilters.scripts.length > 0
                ? 'border-purple-400 ring-2 ring-purple-100'
                : 'border-slate-200'
            }`}>
              <div className="flex items-center gap-2 mb-3">
                <svg className="w-4 h-4 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
                </svg>
                <h3 className="font-semibold text-slate-800 text-sm">Script Type</h3>
                {localFilters.scripts.length > 0 && (
                  <span className="ml-auto text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded-full font-semibold">Active</span>
                )}
              </div>
              {availableScripts.length > 0 ? (
                <div className="space-y-2">
                  {availableScripts.map((script) => (
                    <label
                      key={script}
                      className="flex items-center gap-3 p-2 rounded-md hover:bg-slate-50 cursor-pointer transition-colors group"
                    >
                      <input
                        type="checkbox"
                        checked={localFilters.scripts.includes(script)}
                        onChange={() => handleScriptToggle(script)}
                        className="w-4 h-4 text-purple-600 border-slate-300 rounded focus:ring-2 focus:ring-purple-500 cursor-pointer"
                      />
                      <span className="text-sm text-slate-700 group-hover:text-slate-900 font-medium">
                        {script}
                      </span>
                    </label>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-slate-500 italic">No script types available</p>
              )}
              {localFilters.scripts.length > 0 && (
                <div className="mt-3 pt-3 border-t border-slate-200">
                  <p className="text-xs text-slate-600">
                    Selected: <span className="font-medium">{localFilters.scripts.join(', ')}</span>
                  </p>
                </div>
              )}
            </div>

            {/* Edge Piece Filter */}
            <div className={`bg-white rounded-lg p-4 shadow-sm border-2 transition-all ${
              localFilters.isEdgePiece !== null
                ? 'border-emerald-400 ring-2 ring-emerald-100'
                : 'border-slate-200'
            }`}>
              <div className="flex items-center gap-2 mb-3">
                <svg className="w-4 h-4 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
                <h3 className="font-semibold text-slate-800 text-sm">Edge Piece</h3>
                {localFilters.isEdgePiece !== null && (
                  <span className="ml-auto text-xs bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded-full font-semibold">Active</span>
                )}
              </div>
              <div className="space-y-2">
                <label className="flex items-center gap-3 p-2 rounded-md hover:bg-slate-50 cursor-pointer transition-colors">
                  <input
                    type="radio"
                    name="edgePiece"
                    checked={localFilters.isEdgePiece === null}
                    onChange={() => handleEdgePieceChange(null)}
                    className="w-4 h-4 text-slate-600 border-slate-300 focus:ring-2 focus:ring-slate-500 cursor-pointer"
                  />
                  <span className="text-sm text-slate-700 font-medium">Don't care</span>
                </label>
                <label className="flex items-center gap-3 p-2 rounded-md hover:bg-slate-50 cursor-pointer transition-colors">
                  <input
                    type="radio"
                    name="edgePiece"
                    checked={localFilters.isEdgePiece === true}
                    onChange={() => handleEdgePieceChange(true)}
                    className="w-4 h-4 text-emerald-600 border-slate-300 focus:ring-2 focus:ring-emerald-500 cursor-pointer"
                  />
                  <span className="text-sm text-slate-700 font-medium">Yes</span>
                </label>
                <label className="flex items-center gap-3 p-2 rounded-md hover:bg-slate-50 cursor-pointer transition-colors">
                  <input
                    type="radio"
                    name="edgePiece"
                    checked={localFilters.isEdgePiece === false}
                    onChange={() => handleEdgePieceChange(false)}
                    className="w-4 h-4 text-slate-600 border-slate-300 focus:ring-2 focus:ring-slate-500 cursor-pointer"
                  />
                  <span className="text-sm text-slate-700 font-medium">No</span>
                </label>
              </div>
            </div>

            {/* Top Edge Filter */}
            <div className={`bg-white rounded-lg p-4 shadow-sm border-2 transition-all ${
              localFilters.hasTopEdge !== null
                ? 'border-emerald-400 ring-2 ring-emerald-100'
                : 'border-slate-200'
            }`}>
              <div className="flex items-center gap-2 mb-3">
                <svg className="w-4 h-4 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                </svg>
                <h3 className="font-semibold text-slate-800 text-sm">Top Edge</h3>
                {localFilters.hasTopEdge !== null && (
                  <span className="ml-auto text-xs bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded-full font-semibold">Active</span>
                )}
              </div>
              <div className="space-y-2">
                <label className="flex items-center gap-3 p-2 rounded-md hover:bg-slate-50 cursor-pointer transition-colors">
                  <input
                    type="radio"
                    name="topEdge"
                    checked={localFilters.hasTopEdge === null}
                    onChange={() => handleTopEdgeChange(null)}
                    className="w-4 h-4 text-slate-600 border-slate-300 focus:ring-2 focus:ring-slate-500 cursor-pointer"
                  />
                  <span className="text-sm text-slate-700 font-medium">Don't care</span>
                </label>
                <label className="flex items-center gap-3 p-2 rounded-md hover:bg-slate-50 cursor-pointer transition-colors">
                  <input
                    type="radio"
                    name="topEdge"
                    checked={localFilters.hasTopEdge === true}
                    onChange={() => handleTopEdgeChange(true)}
                    className="w-4 h-4 text-emerald-600 border-slate-300 focus:ring-2 focus:ring-emerald-500 cursor-pointer"
                  />
                  <span className="text-sm text-slate-700 font-medium">Yes</span>
                </label>
                <label className="flex items-center gap-3 p-2 rounded-md hover:bg-slate-50 cursor-pointer transition-colors">
                  <input
                    type="radio"
                    name="topEdge"
                    checked={localFilters.hasTopEdge === false}
                    onChange={() => handleTopEdgeChange(false)}
                    className="w-4 h-4 text-slate-600 border-slate-300 focus:ring-2 focus:ring-slate-500 cursor-pointer"
                  />
                  <span className="text-sm text-slate-700 font-medium">No</span>
                </label>
              </div>
            </div>

            {/* Bottom Edge Filter */}
            <div className={`bg-white rounded-lg p-4 shadow-sm border-2 transition-all ${
              localFilters.hasBottomEdge !== null
                ? 'border-emerald-400 ring-2 ring-emerald-100'
                : 'border-slate-200'
            }`}>
              <div className="flex items-center gap-2 mb-3">
                <svg className="w-4 h-4 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                </svg>
                <h3 className="font-semibold text-slate-800 text-sm">Bottom Edge</h3>
                {localFilters.hasBottomEdge !== null && (
                  <span className="ml-auto text-xs bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded-full font-semibold">Active</span>
                )}
              </div>
              <div className="space-y-2">
                <label className="flex items-center gap-3 p-2 rounded-md hover:bg-slate-50 cursor-pointer transition-colors">
                  <input
                    type="radio"
                    name="bottomEdge"
                    checked={localFilters.hasBottomEdge === null}
                    onChange={() => handleBottomEdgeChange(null)}
                    className="w-4 h-4 text-slate-600 border-slate-300 focus:ring-2 focus:ring-slate-500 cursor-pointer"
                  />
                  <span className="text-sm text-slate-700 font-medium">Don't care</span>
                </label>
                <label className="flex items-center gap-3 p-2 rounded-md hover:bg-slate-50 cursor-pointer transition-colors">
                  <input
                    type="radio"
                    name="bottomEdge"
                    checked={localFilters.hasBottomEdge === true}
                    onChange={() => handleBottomEdgeChange(true)}
                    className="w-4 h-4 text-emerald-600 border-slate-300 focus:ring-2 focus:ring-emerald-500 cursor-pointer"
                  />
                  <span className="text-sm text-slate-700 font-medium">Yes</span>
                </label>
                <label className="flex items-center gap-3 p-2 rounded-md hover:bg-slate-50 cursor-pointer transition-colors">
                  <input
                    type="radio"
                    name="bottomEdge"
                    checked={localFilters.hasBottomEdge === false}
                    onChange={() => handleBottomEdgeChange(false)}
                    className="w-4 h-4 text-slate-600 border-slate-300 focus:ring-2 focus:ring-slate-500 cursor-pointer"
                  />
                  <span className="text-sm text-slate-700 font-medium">No</span>
                </label>
              </div>
            </div>

            {/* Left Edge Filter */}
            <div className={`bg-white rounded-lg p-4 shadow-sm border-2 transition-all ${
              localFilters.hasLeftEdge !== null
                ? 'border-emerald-400 ring-2 ring-emerald-100'
                : 'border-slate-200'
            }`}>
              <div className="flex items-center gap-2 mb-3">
                <svg className="w-4 h-4 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                <h3 className="font-semibold text-slate-800 text-sm">Left Edge</h3>
                {localFilters.hasLeftEdge !== null && (
                  <span className="ml-auto text-xs bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded-full font-semibold">Active</span>
                )}
              </div>
              <div className="space-y-2">
                <label className="flex items-center gap-3 p-2 rounded-md hover:bg-slate-50 cursor-pointer transition-colors">
                  <input
                    type="radio"
                    name="leftEdge"
                    checked={localFilters.hasLeftEdge === null}
                    onChange={() => handleLeftEdgeChange(null)}
                    className="w-4 h-4 text-slate-600 border-slate-300 focus:ring-2 focus:ring-slate-500 cursor-pointer"
                  />
                  <span className="text-sm text-slate-700 font-medium">Don't care</span>
                </label>
                <label className="flex items-center gap-3 p-2 rounded-md hover:bg-slate-50 cursor-pointer transition-colors">
                  <input
                    type="radio"
                    name="leftEdge"
                    checked={localFilters.hasLeftEdge === true}
                    onChange={() => handleLeftEdgeChange(true)}
                    className="w-4 h-4 text-emerald-600 border-slate-300 focus:ring-2 focus:ring-emerald-500 cursor-pointer"
                  />
                  <span className="text-sm text-slate-700 font-medium">Yes</span>
                </label>
                <label className="flex items-center gap-3 p-2 rounded-md hover:bg-slate-50 cursor-pointer transition-colors">
                  <input
                    type="radio"
                    name="leftEdge"
                    checked={localFilters.hasLeftEdge === false}
                    onChange={() => handleLeftEdgeChange(false)}
                    className="w-4 h-4 text-slate-600 border-slate-300 focus:ring-2 focus:ring-slate-500 cursor-pointer"
                  />
                  <span className="text-sm text-slate-700 font-medium">No</span>
                </label>
              </div>
            </div>

            {/* Right Edge Filter */}
            <div className={`bg-white rounded-lg p-4 shadow-sm border-2 transition-all ${
              localFilters.hasRightEdge !== null
                ? 'border-emerald-400 ring-2 ring-emerald-100'
                : 'border-slate-200'
            }`}>
              <div className="flex items-center gap-2 mb-3">
                <svg className="w-4 h-4 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
                <h3 className="font-semibold text-slate-800 text-sm">Right Edge</h3>
                {localFilters.hasRightEdge !== null && (
                  <span className="ml-auto text-xs bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded-full font-semibold">Active</span>
                )}
              </div>
              <div className="space-y-2">
                <label className="flex items-center gap-3 p-2 rounded-md hover:bg-slate-50 cursor-pointer transition-colors">
                  <input
                    type="radio"
                    name="rightEdge"
                    checked={localFilters.hasRightEdge === null}
                    onChange={() => handleRightEdgeChange(null)}
                    className="w-4 h-4 text-slate-600 border-slate-300 focus:ring-2 focus:ring-slate-500 cursor-pointer"
                  />
                  <span className="text-sm text-slate-700 font-medium">Don't care</span>
                </label>
                <label className="flex items-center gap-3 p-2 rounded-md hover:bg-slate-50 cursor-pointer transition-colors">
                  <input
                    type="radio"
                    name="rightEdge"
                    checked={localFilters.hasRightEdge === true}
                    onChange={() => handleRightEdgeChange(true)}
                    className="w-4 h-4 text-emerald-600 border-slate-300 focus:ring-2 focus:ring-emerald-500 cursor-pointer"
                  />
                  <span className="text-sm text-slate-700 font-medium">Yes</span>
                </label>
                <label className="flex items-center gap-3 p-2 rounded-md hover:bg-slate-50 cursor-pointer transition-colors">
                  <input
                    type="radio"
                    name="rightEdge"
                    checked={localFilters.hasRightEdge === false}
                    onChange={() => handleRightEdgeChange(false)}
                    className="w-4 h-4 text-slate-600 border-slate-300 focus:ring-2 focus:ring-slate-500 cursor-pointer"
                  />
                  <span className="text-sm text-slate-700 font-medium">No</span>
                </label>
              </div>
            </div>

            {/* Circle Detection Filter */}
            <div className={`bg-white rounded-lg p-4 shadow-sm border-2 transition-all ${
              localFilters.hasCircle !== null
                ? 'border-cyan-400 ring-2 ring-cyan-100'
                : 'border-slate-200'
            }`}>
              <div className="flex items-center gap-2 mb-3">
                <svg className="w-4 h-4 text-cyan-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <circle cx="12" cy="12" r="9" strokeWidth={2} />
                </svg>
                <h3 className="font-semibold text-slate-800 text-sm">Has Circle</h3>
                {localFilters.hasCircle !== null && (
                  <span className="ml-auto text-xs bg-cyan-100 text-cyan-700 px-2 py-0.5 rounded-full font-semibold">Active</span>
                )}
              </div>
              <div className="space-y-2">
                <label className="flex items-center gap-3 p-2 rounded-md hover:bg-slate-50 cursor-pointer transition-colors">
                  <input
                    type="radio"
                    name="hasCircle"
                    checked={localFilters.hasCircle === null}
                    onChange={() => handleCircleChange(null)}
                    className="w-4 h-4 text-slate-600 border-slate-300 focus:ring-2 focus:ring-slate-500 cursor-pointer"
                  />
                  <span className="text-sm text-slate-700 font-medium">Don't care</span>
                </label>
                <label className="flex items-center gap-3 p-2 rounded-md hover:bg-slate-50 cursor-pointer transition-colors">
                  <input
                    type="radio"
                    name="hasCircle"
                    checked={localFilters.hasCircle === true}
                    onChange={() => handleCircleChange(true)}
                    className="w-4 h-4 text-cyan-600 border-slate-300 focus:ring-2 focus:ring-cyan-500 cursor-pointer"
                  />
                  <span className="text-sm text-slate-700 font-medium">Yes</span>
                </label>
                <label className="flex items-center gap-3 p-2 rounded-md hover:bg-slate-50 cursor-pointer transition-colors">
                  <input
                    type="radio"
                    name="hasCircle"
                    checked={localFilters.hasCircle === false}
                    onChange={() => handleCircleChange(false)}
                    className="w-4 h-4 text-slate-600 border-slate-300 focus:ring-2 focus:ring-slate-500 cursor-pointer"
                  />
                  <span className="text-sm text-slate-700 font-medium">No</span>
                </label>
              </div>
            </div>

            {/* Custom Filters */}
            <div className={`bg-white rounded-lg p-4 shadow-sm border-2 transition-all ${
              hasCustomFilters ? 'border-amber-400 ring-2 ring-amber-100' : 'border-slate-200'
            }`}>
              <div className="flex items-center gap-2 mb-3">
                <svg className="w-4 h-4 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 012-2h2a2 2 0 012 2M9 5h6" />
                </svg>
                <h3 className="font-semibold text-slate-800 text-sm">Custom Filters</h3>
                {hasCustomFilters && (
                  <span className="ml-auto text-xs bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full font-semibold">Active</span>
                )}
              </div>

              {customFilters.length === 0 && !isAddingCustom && (
                <p className="text-sm text-slate-500 italic">No custom filters yet.</p>
              )}

              {customFilters.length > 0 && (
                <div className="space-y-3">
                  {customFilters.map((filter) => {
                    const value = localFilters.custom?.[filter.filterKey] ?? '';
                    const isActive = value !== undefined && value !== null && value !== '';

                    return (
                      <div key={filter.filterKey} className="space-y-1">
                        <div className="flex items-center justify-between gap-2">
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-semibold text-slate-700">{filter.label}</span>
                            {isActive && (
                              <span className="text-[10px] uppercase tracking-wide text-amber-600">Active</span>
                            )}
                          </div>
                          <div className="flex items-center gap-2">
                            {filter.type === 'dropdown' && (
                              <button
                                onClick={() => handleStartEditOptions(filter)}
                                className="text-[10px] font-semibold text-amber-700 hover:text-amber-800 uppercase tracking-wide"
                                title="Edit options"
                              >
                                Edit Options
                              </button>
                            )}
                            <button
                              onClick={() => handleDeleteCustomFilter(filter)}
                              className="text-[10px] font-semibold text-red-600 hover:text-red-700 uppercase tracking-wide"
                              title="Delete filter"
                            >
                              Delete
                            </button>
                          </div>
                        </div>
                        {editingOptionsId === filter.id && filter.type === 'dropdown' && (
                          <div className="space-y-2 rounded-md border border-amber-200 bg-amber-50/60 p-2">
                            <label className="block text-xs text-slate-600">Options (comma or new line)</label>
                            <textarea
                              value={optionsDraft}
                              onChange={(e) => setOptionsDraft(e.target.value)}
                              rows={3}
                              className="w-full px-2 py-1.5 border border-amber-200 rounded-md text-xs focus:ring-2 focus:ring-amber-500 focus:border-amber-500 outline-none transition-all bg-white"
                            />
                            {optionsError && (
                              <div className="text-xs text-red-600 bg-red-50 px-2 py-1 rounded">
                                {optionsError}
                              </div>
                            )}
                            <div className="flex items-center gap-2">
                              <button
                                onClick={() => handleSaveOptions(filter)}
                                disabled={isSavingOptions}
                                className="px-2.5 py-1.5 rounded-md text-xs font-semibold text-white bg-amber-600 hover:bg-amber-700 transition-colors disabled:opacity-50"
                              >
                                {isSavingOptions ? 'Saving...' : 'Save'}
                              </button>
                              <button
                                onClick={handleCancelEditOptions}
                                className="px-2.5 py-1.5 rounded-md text-xs font-semibold text-slate-600 bg-white border border-slate-200 hover:bg-slate-50 transition-colors"
                              >
                                Cancel
                              </button>
                            </div>
                          </div>
                        )}
                        {filter.type === 'dropdown' ? (
                          <select
                            value={value}
                            onChange={(e) => handleCustomValueChange(filter.filterKey, e.target.value)}
                            className="w-full px-3 py-2 border border-slate-300 rounded-md text-sm focus:ring-2 focus:ring-amber-500 focus:border-amber-500 outline-none transition-all bg-white"
                          >
                            <option value="">Any</option>
                            {(filter.options || []).map((option) => (
                              <option key={option} value={option}>
                                {option}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <input
                            type="text"
                            value={value}
                            onChange={(e) => handleCustomValueChange(filter.filterKey, e.target.value)}
                            placeholder="Exact match"
                            className="w-full px-3 py-2 border border-slate-300 rounded-md text-sm focus:ring-2 focus:ring-amber-500 focus:border-amber-500 outline-none transition-all"
                          />
                        )}
                      </div>
                    );
                  })}
                </div>
              )}

              <div className="mt-4 border-t border-slate-200 pt-3">
                {!isAddingCustom ? (
                  <button
                    onClick={() => setIsAddingCustom(true)}
                    className="w-full px-3 py-2 rounded-md text-sm font-semibold text-amber-700 bg-amber-50 hover:bg-amber-100 transition-colors"
                  >
                    Add Custom Filter
                  </button>
                ) : (
                  <div className="space-y-3">
                    <div>
                      <label className="block text-xs text-slate-600 mb-1">Filter name</label>
                      <input
                        type="text"
                        value={newFilterLabel}
                        onChange={(e) => setNewFilterLabel(e.target.value)}
                        className="w-full px-3 py-2 border border-slate-300 rounded-md text-sm focus:ring-2 focus:ring-amber-500 focus:border-amber-500 outline-none transition-all"
                        placeholder="e.g. Material"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="block text-xs text-slate-600">Input type</label>
                      <div className="flex items-center gap-3 text-sm">
                        <label className="flex items-center gap-2">
                          <input
                            type="radio"
                            name="customFilterType"
                            checked={newFilterType === 'dropdown'}
                            onChange={() => setNewFilterType('dropdown')}
                            className="w-4 h-4 text-amber-600 border-slate-300 focus:ring-2 focus:ring-amber-500 cursor-pointer"
                          />
                          Dropdown
                        </label>
                        <label className="flex items-center gap-2">
                          <input
                            type="radio"
                            name="customFilterType"
                            checked={newFilterType === 'text'}
                            onChange={() => setNewFilterType('text')}
                            className="w-4 h-4 text-amber-600 border-slate-300 focus:ring-2 focus:ring-amber-500 cursor-pointer"
                          />
                          Text match
                        </label>
                      </div>
                    </div>
                    {newFilterType === 'dropdown' && (
                      <div>
                        <label className="block text-xs text-slate-600 mb-1">Options (comma or new line)</label>
                        <textarea
                          value={newFilterOptions}
                          onChange={(e) => setNewFilterOptions(e.target.value)}
                          rows={3}
                          className="w-full px-3 py-2 border border-slate-300 rounded-md text-sm focus:ring-2 focus:ring-amber-500 focus:border-amber-500 outline-none transition-all"
                          placeholder="paper, parchment, palm"
                        />
                      </div>
                    )}
                    {newFilterError && (
                      <div className="text-xs text-red-600 bg-red-50 px-2 py-1 rounded">
                        {newFilterError}
                      </div>
                    )}
                    <div className="flex items-center gap-2">
                      <button
                        onClick={handleCreateCustom}
                        disabled={isCreatingCustom}
                        className="flex-1 px-3 py-2 rounded-md text-sm font-semibold text-white bg-amber-600 hover:bg-amber-700 transition-colors disabled:opacity-50"
                      >
                        {isCreatingCustom ? 'Creating...' : 'Create Filter'}
                      </button>
                      <button
                        onClick={() => {
                          setIsAddingCustom(false);
                          setNewFilterError(null);
                        }}
                        className="px-3 py-2 rounded-md text-sm font-semibold text-slate-600 bg-slate-100 hover:bg-slate-200 transition-colors"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="p-4 bg-white sticky bottom-0 space-y-2.5" style={{
            borderTop: '1px solid rgba(214, 211, 209, 0.4)'
          }}>
            <button
              onClick={handleApply}
              className="w-full px-4 py-3 text-white rounded-xl font-semibold shadow-lg hover:shadow-xl transition-all duration-200 flex items-center justify-center gap-2 font-body"
              style={{
                background: 'linear-gradient(135deg, #ea580c 0%, #c2410c 100%)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'linear-gradient(135deg, #f97316 0%, #ea580c 100%)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'linear-gradient(135deg, #ea580c 0%, #c2410c 100%)';
              }}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
              Apply Filters
            </button>
            <button
              onClick={handleReset}
              disabled={!hasActiveFilters}
              className="w-full px-4 py-2.5 rounded-xl font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 font-body"
              style={{
                background: 'rgba(231, 229, 228, 0.8)',
                color: '#292524'
              }}
              onMouseEnter={(e) => {
                if (hasActiveFilters) {
                  e.currentTarget.style.background = 'rgba(214, 211, 209, 0.9)';
                }
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'rgba(231, 229, 228, 0.8)';
              }}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Reset Filters
            </button>
          </div>

          {/* Resize Handle */}
          <div
            onMouseDown={handleMouseDown}
            className={`absolute top-0 left-0 w-1 h-full cursor-col-resize hover:bg-blue-500 transition-colors group ${
              isResizing ? 'bg-blue-500' : 'bg-transparent'
            }`}
            title="Drag to resize"
          >
            <div className="absolute top-1/2 left-0 -translate-y-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
              <div className="bg-blue-500 text-white rounded-full p-1 shadow-lg">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l4-4 4 4m0 6l-4 4-4-4" />
                </svg>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default FilterPanel;
