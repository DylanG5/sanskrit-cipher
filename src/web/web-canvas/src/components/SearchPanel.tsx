import React, { useState, useCallback, useEffect } from 'react';
import {
    SearchQuery,
    DEFAULT_SEARCH_QUERY,
    SearchResultPage,
    EMPTY_SEARCH_RESULT_PAGE,
    ExportFormat,
    ValidationError,
} from '../types/search';
import { ManuscriptFragment } from '../types/fragment';
import SearchResultsTable from './SearchResultsTable';
import PaginationControls from './PaginationControls';
import ExportButton from './ExportButton';
import {
    executeSearch,
    getAvailableScripts,
} from '../services/search-service';
import { exportResults } from '../services/export-service';

interface SearchPanelProps {
    onFragmentSelect?: (fragment: ManuscriptFragment) => void;
}

const SearchPanel: React.FC<SearchPanelProps> = ({ onFragmentSelect }) => {
    // State variables per MIS specification
    const [currentQuery, setCurrentQuery] = useState<SearchQuery>(DEFAULT_SEARCH_QUERY);
    const [currentPage, setCurrentPage] = useState<number>(0);
    const [results, setResults] = useState<SearchResultPage>(EMPTY_SEARCH_RESULT_PAGE);

    // UI state
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [availableScripts, setAvailableScripts] = useState<string[]>([]);
    const [hasSearched, setHasSearched] = useState(false);

    // Form state (separate from committed query)
    const [formQuery, setFormQuery] = useState<SearchQuery>(DEFAULT_SEARCH_QUERY);

    // Load available scripts on mount
    useEffect(() => {
        const loadScripts = async () => {
            const scripts = await getAvailableScripts();
            setAvailableScripts(scripts);
        };
        loadScripts();
    }, []);

    // ============================================================================
    // MIS Access Routines
    // ============================================================================

    const submitQuery = useCallback(async (q: SearchQuery) => {
        setError(null);
        setIsLoading(true);
        setHasSearched(true);

        try {
            const searchResults = await executeSearch(q, 0);
            setCurrentQuery(q);
            setCurrentPage(0);
            setResults(searchResults);
        } catch (err) {
            if (err instanceof ValidationError) {
                setError(`Validation Error: ${err.message}`);
            } else {
                setError(err instanceof Error ? err.message : 'Search failed');
            }
            setResults(EMPTY_SEARCH_RESULT_PAGE);
        } finally {
            setIsLoading(false);
        }
    }, []);

    /**
     * nextPage() - MIS access routine
     */
    const nextPage = useCallback(async () => {
        if (!results.hasNextPage || isLoading) return;

        setError(null);
        setIsLoading(true);

        try {
            const nextResults = await executeSearch(currentQuery, currentPage + 1);
            setCurrentPage(currentPage + 1);
            setResults(nextResults);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load next page');
        } finally {
            setIsLoading(false);
        }
    }, [currentQuery, currentPage, results.hasNextPage, isLoading]);

    /**
     * Previous page navigation
     */
    const previousPage = useCallback(async () => {
        if (!results.hasPreviousPage || isLoading) return;

        setError(null);
        setIsLoading(true);

        try {
            const prevResults = await executeSearch(currentQuery, currentPage - 1);
            setCurrentPage(currentPage - 1);
            setResults(prevResults);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load previous page');
        } finally {
            setIsLoading(false);
        }
    }, [currentQuery, currentPage, results.hasPreviousPage, isLoading]);

    /**
     * Go to specific page
     */
    const goToPage = useCallback(async (page: number) => {
        if (page === currentPage || isLoading) return;

        setError(null);
        setIsLoading(true);

        try {
            const pageResults = await executeSearch(currentQuery, page);
            setCurrentPage(page);
            setResults(pageResults);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load page');
        } finally {
            setIsLoading(false);
        }
    }, [currentQuery, currentPage, isLoading]);

    /**
     * exportResults(format) - MIS access routine
     */
    const handleExport = useCallback(async (format: ExportFormat) => {
        await exportResults(currentQuery, format);
    }, [currentQuery]);

    const handleSearchSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        submitQuery(formQuery);
    };

    const handleReset = () => {
        setFormQuery(DEFAULT_SEARCH_QUERY);
        setCurrentQuery(DEFAULT_SEARCH_QUERY);
        setResults(EMPTY_SEARCH_RESULT_PAGE);
        setCurrentPage(0);
        setHasSearched(false);
        setError(null);
    };

    const handleScriptToggle = (script: string) => {
        setFormQuery((prev) => ({
            ...prev,
            scripts: prev.scripts.includes(script)
                ? prev.scripts.filter((s) => s !== script)
                : [...prev.scripts, script],
        }));
    };

    return (
        <div className="flex flex-col h-full">
            {/* Search Form */}
            <form onSubmit={handleSearchSubmit} className="flex-shrink-0">
                <div
                    className="p-6 border-b space-y-5"
                    style={{
                        background: 'white',
                        borderColor: 'var(--color-neutral-200)',
                    }}
                >
                    {/* Search Input */}
                    <div>
                        <label
                            className="block text-sm font-semibold mb-2"
                            style={{ color: 'var(--color-neutral-700)' }}
                        >
                            Search Fragments
                        </label>
                        <div className="relative">
                            <svg
                                className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5"
                                style={{ color: 'var(--color-neutral-400)' }}
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                                strokeWidth={2}
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                                />
                            </svg>
                            <input
                                type="text"
                                placeholder="Search by fragment ID..."
                                value={formQuery.searchText || ''}
                                onChange={(e) =>
                                    setFormQuery((prev) => ({ ...prev, searchText: e.target.value }))
                                }
                                className="w-full pl-12 pr-4 py-3 rounded-xl border text-sm transition-all duration-200"
                                style={{
                                    borderColor: 'var(--color-neutral-300)',
                                    color: 'var(--color-neutral-800)',
                                }}
                                onFocus={(e) => {
                                    e.target.style.borderColor = 'var(--color-primary-400)';
                                    e.target.style.boxShadow = '0 0 0 3px rgba(249, 115, 22, 0.1)';
                                }}
                                onBlur={(e) => {
                                    e.target.style.borderColor = 'var(--color-neutral-300)';
                                    e.target.style.boxShadow = 'none';
                                }}
                            />
                        </div>
                    </div>

                    {/* Filters Row */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {/* Line Count Range */}
                        <div>
                            <label
                                className="block text-sm font-semibold mb-2"
                                style={{ color: 'var(--color-neutral-700)' }}
                            >
                                Line Count
                            </label>
                            <div className="flex items-center gap-2">
                                <input
                                    type="number"
                                    placeholder="Min"
                                    min="0"
                                    value={formQuery.lineCountMin ?? ''}
                                    onChange={(e) =>
                                        setFormQuery((prev) => ({
                                            ...prev,
                                            lineCountMin: e.target.value ? parseInt(e.target.value, 10) : undefined,
                                        }))
                                    }
                                    className="flex-1 px-3 py-2 rounded-lg border text-sm"
                                    style={{ borderColor: 'var(--color-neutral-300)' }}
                                />
                                <span style={{ color: 'var(--color-neutral-400)' }}>–</span>
                                <input
                                    type="number"
                                    placeholder="Max"
                                    min="0"
                                    value={formQuery.lineCountMax ?? ''}
                                    onChange={(e) =>
                                        setFormQuery((prev) => ({
                                            ...prev,
                                            lineCountMax: e.target.value ? parseInt(e.target.value, 10) : undefined,
                                        }))
                                    }
                                    className="flex-1 px-3 py-2 rounded-lg border text-sm"
                                    style={{ borderColor: 'var(--color-neutral-300)' }}
                                />
                            </div>
                        </div>

                        {/* Edge Piece Filter */}
                        <div>
                            <label
                                className="block text-sm font-semibold mb-2"
                                style={{ color: 'var(--color-neutral-700)' }}
                            >
                                Edge Piece
                            </label>
                            <select
                                value={formQuery.isEdgePiece === null ? '' : formQuery.isEdgePiece.toString()}
                                onChange={(e) =>
                                    setFormQuery((prev) => ({
                                        ...prev,
                                        isEdgePiece:
                                            e.target.value === '' ? null : e.target.value === 'true',
                                    }))
                                }
                                className="w-full px-3 py-2 rounded-lg border text-sm"
                                style={{
                                    borderColor: 'var(--color-neutral-300)',
                                    color: 'var(--color-neutral-700)',
                                }}
                            >
                                <option value="">Any</option>
                                <option value="true">Yes</option>
                                <option value="false">No</option>
                            </select>
                        </div>

                        {/* Script Types */}
                        <div>
                            <label
                                className="block text-sm font-semibold mb-2"
                                style={{ color: 'var(--color-neutral-700)' }}
                            >
                                Script Types
                            </label>
                            {availableScripts.length > 0 ? (
                                <div className="flex flex-wrap gap-2">
                                    {availableScripts.slice(0, 4).map((script) => (
                                        <button
                                            key={script}
                                            type="button"
                                            onClick={() => handleScriptToggle(script)}
                                            className="px-2.5 py-1 rounded-full text-xs font-medium transition-all duration-150"
                                            style={{
                                                background: formQuery.scripts.includes(script)
                                                    ? 'var(--color-primary-600)'
                                                    : 'var(--color-neutral-100)',
                                                color: formQuery.scripts.includes(script)
                                                    ? 'white'
                                                    : 'var(--color-neutral-600)',
                                            }}
                                        >
                                            {script}
                                        </button>
                                    ))}
                                </div>
                            ) : (
                                <span
                                    className="text-sm"
                                    style={{ color: 'var(--color-neutral-400)' }}
                                >
                                    No script types available
                                </span>
                            )}
                        </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex items-center justify-between pt-2">
                        <button
                            type="button"
                            onClick={handleReset}
                            className="px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-150"
                            style={{ color: 'var(--color-neutral-600)' }}
                            onMouseEnter={(e) => {
                                e.currentTarget.style.background = 'var(--color-neutral-100)';
                            }}
                            onMouseLeave={(e) => {
                                e.currentTarget.style.background = 'transparent';
                            }}
                        >
                            Reset Filters
                        </button>

                        <div className="flex items-center gap-3">
                            <ExportButton
                                onExport={handleExport}
                                disabled={!hasSearched || results.totalCount === 0}
                                resultCount={results.totalCount}
                            />
                            <button
                                type="submit"
                                disabled={isLoading}
                                className="flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-semibold text-white transition-all duration-200 disabled:cursor-not-allowed"
                                style={{
                                    background: isLoading
                                        ? 'var(--color-neutral-400)'
                                        : 'linear-gradient(135deg, var(--color-primary-600), var(--color-primary-700))',
                                }}
                                onMouseEnter={(e) => {
                                    if (!isLoading) {
                                        e.currentTarget.style.background =
                                            'linear-gradient(135deg, var(--color-primary-500), var(--color-primary-600))';
                                        e.currentTarget.style.transform = 'translateY(-1px)';
                                    }
                                }}
                                onMouseLeave={(e) => {
                                    if (!isLoading) {
                                        e.currentTarget.style.background =
                                            'linear-gradient(135deg, var(--color-primary-600), var(--color-primary-700))';
                                        e.currentTarget.style.transform = 'translateY(0)';
                                    }
                                }}
                            >
                                {isLoading ? (
                                    <>
                                        <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
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
                                        Searching...
                                    </>
                                ) : (
                                    <>
                                        <svg
                                            className="w-4 h-4"
                                            fill="none"
                                            stroke="currentColor"
                                            viewBox="0 0 24 24"
                                            strokeWidth={2}
                                        >
                                            <path
                                                strokeLinecap="round"
                                                strokeLinejoin="round"
                                                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                                            />
                                        </svg>
                                        Search
                                    </>
                                )}
                            </button>
                        </div>
                    </div>

                    {/* Error Display */}
                    {error && (
                        <div
                            className="p-3 rounded-lg text-sm"
                            style={{
                                background: 'rgba(239, 68, 68, 0.1)',
                                color: 'var(--color-error)',
                            }}
                        >
                            {error}
                        </div>
                    )}
                </div>
            </form>

            {/* Results Section */}
            <div
                className="flex-1 overflow-auto p-6"
                style={{ background: 'var(--color-neutral-50)' }}
            >
                {/* Results Header */}
                {hasSearched && (
                    <div className="flex items-center justify-between mb-4">
                        <h3
                            className="text-lg font-semibold"
                            style={{ color: 'var(--color-neutral-800)' }}
                        >
                            Search Results
                            {results.totalCount > 0 && (
                                <span
                                    className="ml-2 text-sm font-normal"
                                    style={{ color: 'var(--color-neutral-500)' }}
                                >
                                    ({results.totalCount.toLocaleString()} found)
                                </span>
                            )}
                        </h3>
                    </div>
                )}

                {/* Results Table */}
                <div
                    className="rounded-xl border overflow-hidden"
                    style={{
                        background: 'white',
                        borderColor: 'var(--color-neutral-200)',
                    }}
                >
                    {hasSearched ? (
                        <>
                            <div className="p-4">
                                <SearchResultsTable
                                    results={results.items}
                                    isLoading={isLoading}
                                    onFragmentClick={onFragmentSelect}
                                />
                            </div>

                            <PaginationControls
                                currentPage={results.currentPage}
                                totalPages={results.totalPages}
                                totalCount={results.totalCount}
                                hasNextPage={results.hasNextPage}
                                hasPreviousPage={results.hasPreviousPage}
                                isLoading={isLoading}
                                onNextPage={nextPage}
                                onPreviousPage={previousPage}
                                onGoToPage={goToPage}
                            />
                        </>
                    ) : (
                        <div className="flex flex-col items-center justify-center py-20 text-center">
                            <div
                                className="w-24 h-24 rounded-2xl flex items-center justify-center mb-6"
                                style={{ background: 'var(--color-primary-100)' }}
                            >
                                <svg
                                    className="w-12 h-12"
                                    style={{ color: 'var(--color-primary-600)' }}
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                    strokeWidth={1.5}
                                >
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z"
                                    />
                                </svg>
                            </div>
                            <h3
                                className="text-xl font-semibold mb-2"
                                style={{ color: 'var(--color-neutral-800)' }}
                            >
                                Search the Fragment Database
                            </h3>
                            <p
                                className="text-sm max-w-sm"
                                style={{ color: 'var(--color-neutral-500)' }}
                            >
                                Use the filters above to search for manuscript fragments by ID, script type,
                                line count, and more.
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default SearchPanel;
